"""Audit saved model outputs offline. Run from any directory; no model calls."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from evidenceswitch.generate import generate_cases
from evidenceswitch.io import parse_json, read_jsonl
from evidenceswitch.score import compare, score
from evidenceswitch.workflow import prepare


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def indexed(rows):
    result = {(row["id"], row["condition"]): row for row in rows}
    check(len(result) == len(rows), "Duplicate timing/trace row")
    return result


def verify_phase(phase, seed, groups, frozen, audit):
    cases = read_jsonl(HERE / f"{phase}-cases.jsonl")
    predictions = {name: read_jsonl(HERE / f"{phase}-{name}.jsonl")
                   for name in ("baseline", "candidate")}
    saved = read_json(f"{phase}-analysis.json")
    check(cases == generate_cases(seed=seed, groups=groups), "Generated cases differ")
    prompts = [{"id": case["id"], "messages": case["messages"]} for case in cases]
    check(digest(prompts) == frozen["prompt_hashes"][phase], "Frozen prompt hash differs")
    check(all(len(rows) == len(cases) for rows in predictions.values()), "Incomplete responses")
    comparison = compare(cases, predictions["baseline"], predictions["candidate"])
    check(comparison == saved["comparison"], "Saved scores or changed IDs differ")

    timing_rows = read_jsonl(HERE / f"{phase}-timings.jsonl")
    timings = indexed(timing_rows)
    traces = indexed(read_jsonl(HERE / f"{phase}-traces.jsonl"))
    keys = {(case["id"], name) for case in cases for name in predictions}
    check(set(timings) == keys == set(traces), "Incomplete timings or traces")
    extra_count = missing_count = 0
    extra_details = []
    for case in cases:
        messages = case["messages"]
        content = messages[1]["content"]
        question = content.splitlines()[0]
        entity = re.search(r"Nuvora-[0-9a-f]+", question).group()
        field = re.search(r"signal_code|route_token|archive_label|beacon_key", question).group()
        records = parse_json(content[content.index("["):])
        matching = [r for r in records if r["entity"] == entity and r["field"] == field]
        winner = max(matching, key=lambda r: r["revision"]) if matching else None
        expected = {"answer": winner["value"] if winner else None,
                    "citations": [winner["source_id"]] if winner else []}
        check(expected == case["expected"], "Independently computed answer key differs")

        prepared = prepare(messages)
        check(prepared.extraction_messages[0]["content"] == frozen["extract_instructions"][case["language"]],
              "Production extraction instruction differs from frozen experiment")
        trace = traces[(case["id"], "candidate")]
        check(trace["extraction_valid"] is True, "Saved extraction was invalid")
        final_messages = prepared.finish(trace["extraction_response"])
        selected = parse_json(trace["extraction_response"])["records"]
        relevant_ids = {r["source_id"] for r in matching}
        selected_ids = {r["source_id"] for r in selected}
        extra, missing = selected_ids - relevant_ids, relevant_ids - selected_ids
        extra_count += bool(extra)
        missing_count += bool(missing)
        if extra or missing:
            extra_details.append({"id": case["id"], "extra": sorted(extra), "missing": sorted(missing)})

        for name in predictions:
            row = timings[(case["id"], name)]
            check(row["original_messages_sha256"] == digest(messages), "Original input hash differs")
            expected_messages = [messages] if name == "baseline" else [prepared.extraction_messages, final_messages]
            check(len(row["calls"]) == len(expected_messages), "Unexpected model call count")
            for call, sent in zip(row["calls"], expected_messages):
                check(call["messages_sha256"] == digest(sent), "Production model prompt differs from measured prompt")
                check(call["model"] == frozen["model"]["name"], "Model name differs")
                check(call["done"] and call["done_reason"] == "stop", "Incomplete model response")
                check(not call["thinking_returned"] and call["thinking_chars"] == 0, "Unexpected thinking output")
                check(call["eval_count"] <= frozen["options"]["num_predict"], "Output exceeded saved cap")

    counts = {"cases": len(cases), "with_extra_records": extra_count,
              "with_missing_matching_records": missing_count, "invalid_extraction": 0}
    check(counts == audit[phase]["counts"], "Extraction audit counts differ")
    normalized = lambda rows: {r["id"]: (set(r["extra"]), set(r["missing"])) for r in rows}
    check(normalized(extra_details) == normalized(audit[phase]["details"]), "Extraction audit details differ")

    paired = {}
    for name, rows in predictions.items():
        pairs = defaultdict(list)
        for result in score(cases, rows)["results"]:
            pairs[result["pair_id"]].append(result["correct"])
        paired[name] = {pair: all(values) for pair, values in pairs.items()}
        condition_rows = [row for row in timing_rows if row["condition"] == name]
        calls = [call for row in condition_rows for call in row["calls"]]
        runtime = {
            "cases": len(condition_rows), "requests": len(calls), "all_done": all(c["done"] for c in calls),
            "stop_reasons": dict(Counter(c["done_reason"] for c in calls)),
            "total_wall_seconds": sum(row["wall_seconds"] for row in condition_rows),
            "median_wall_seconds": statistics.median(row["wall_seconds"] for row in condition_rows),
            "median_inference_seconds_excluding_load": statistics.median(
                sum((c["total_duration"] - c["load_duration"]) / 1e9 for c in row["calls"])
                for row in condition_rows),
            "generated_tokens": sum(c["eval_count"] for c in calls),
            "max_generated_tokens_per_call": max(c["eval_count"] for c in calls),
            "prompt_tokens": sum(c["prompt_eval_count"] for c in calls),
            "max_prompt_tokens_per_call": max(c["prompt_eval_count"] for c in calls),
            "thinking_returned_count": sum(c["thinking_returned"] for c in calls),
        }
        check(runtime == saved["runtime"][name], "Saved timing/token summary differs")

    transitions = Counter((paired["baseline"][p], paired["candidate"][p]) for p in paired["baseline"])
    fixed, regressed = transitions[(False, True)], transitions[(True, False)]
    discordant = fixed + regressed
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(fixed, regressed) + 1)) / 2 ** discordant) if discordant else 1.0
    check(saved["pair_transitions"] == {"both_correct": transitions[(True, True)],
          "two_stage_only_correct": fixed, "baseline_only_correct": regressed,
          "both_incorrect": transitions[(False, False)]}, "Pair transitions differ")
    check(saved["pair_exact_mcnemar_two_sided_p"] == p, "Pair p-value differs")
    print(f"{phase}: {len(cases)} cases; {len(comparison['fixed_ids'])} fixes, "
          f"{len(comparison['regressed_ids'])} regressions; scores, extraction, prompts and costs verified")
    return {case["id"] for case in cases}


def main():
    frozen, audit = read_json("frozen.json"), read_json("extraction-audit.json")
    primary = verify_phase("holdout", 505, 3, frozen, audit)
    confirmation = verify_phase("confirmation", 606, 1, frozen, audit)
    check(not primary & confirmation, "Cases reused between phases")
    check(len(primary | confirmation) == audit["fresh_unique_cases"], "Unique case count differs")
    print("PASS: 72 saved cases and 216 request prompt hashes verified offline. No model requests sent.")
    smoke = HERE / "integration-smoke"
    cases = read_jsonl(HERE.parents[1] / "examples/mini/cases.jsonl")
    predictions = {name: read_jsonl(smoke / f"{name}.jsonl") for name in ("direct", "two-stage")}
    comparison = compare(cases, predictions["direct"], predictions["two-stage"])
    check(comparison == read_json("integration-smoke/comparison.json"), "Integration smoke scores differ")
    check(len(cases) == 6 and comparison["baseline"]["strict_accuracy"]
          == comparison["candidate"]["strict_accuracy"] == 1.0, "Integration smoke result changed")
    for name in predictions:
        traces = {row["id"]: row for row in read_jsonl(smoke / f"{name}-trace.jsonl")}
        check(set(traces) == {case["id"] for case in cases}, "Incomplete integration smoke traces")
        for case in cases:
            row = traces[case["id"]]
            prepared = prepare(case["messages"])
            messages = ([case["messages"]] if name == "direct" else
                        [prepared.extraction_messages, prepared.finish(row["extraction_response"])])
            check([call["messages_sha256"] for call in row["calls"]] == [digest(m) for m in messages],
                  "Integration smoke prompt hashes differ")
            check(all(call["done"] and call["done_reason"] == "stop" and not call["thinking_returned"]
                      for call in row["calls"]), "Incomplete integration smoke calls")
    print("PASS: packaged CLI smoke evidence verified (6 existing cases, both strategies 6/6).")


if __name__ == "__main__":
    main()
