"""Offline audit and paired analysis of the frozen local Llama comparison.

Run: python analyze.py [dev|dev-schema|holdout] [--out new-file.json].
No model, network, or paid service is used. Existing output is never replaced.
"""

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from evidenceswitch.generate import generate_cases
from evidenceswitch.io import parse_json
from evidenceswitch.score import compare, score
from evidenceswitch.workflow import prepare


STRATEGIES = ("direct", "two-stage", "two-stage-ids")
HOLDOUT_STRATEGIES = STRATEGIES + ("two-stage-ids-schema",)
SETS = {"dev": (707, 1), "dev-schema": (707, 1), "holdout": (808, 10)}
QUESTION_PATTERNS = {
    "en": r"What is the field (?P<field>\w+) for entity (?P<entity>Nuvora-[0-9a-f]{12})\?",
    "de": r"Welchen Wert hat das Feld (?P<field>\w+) für die Entität (?P<entity>Nuvora-[0-9a-f]{12})\?",
    "es": r"¿Qué valor tiene el campo (?P<field>\w+) para la entidad (?P<entity>Nuvora-[0-9a-f]{12})\?",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [parse_json(line) for line in handle if line.strip()]


def digest(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False,
                                    sort_keys=True).encode("utf-8")).hexdigest()


def records_from(messages):
    question, payload = messages[1]["content"].split("\n\n", 1)
    _, raw_records = payload.split(":\n", 1)
    return question, parse_json(raw_records)


def gold_from_sources(case):
    """Recompute the answer from the question and records, not stored gold."""
    question, records = records_from(case["messages"])
    match = re.fullmatch(QUESTION_PATTERNS[case["language"]], question)
    require(match is not None, "Question does not match the audited template")
    matching = [record for record in records
                if record["entity"] == match["entity"] and record["field"] == match["field"]]
    if not matching:
        return {"answer": None, "citations": []}, set()
    winner = max(matching, key=lambda record: record["revision"])
    require(sum(record["revision"] == winner["revision"] for record in matching) == 1,
            "Ambiguous highest revision")
    return {"answer": winner["value"], "citations": [winner["source_id"]]}, {
        record["source_id"] for record in matching}


def paired(report):
    groups = defaultdict(list)
    for row in report["results"]:
        groups[row["pair_id"]].append(row["correct"])
    return {key: all(values) for key, values in groups.items()}


def paired_comparison(cases, predictions, reports, old, new):
    result = compare(cases, predictions[old], predictions[new])
    before, after = paired(reports[old]), paired(reports[new])
    fixed = [key for key in before if not before[key] and after[key]]
    regressed = [key for key in before if before[key] and not after[key]]
    b, c = len(fixed), len(regressed)
    n = b + c
    p_value = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    result.update(pair_fixed_ids=fixed, pair_regressed_ids=regressed,
                  pair_both_correct=sum(before[key] and after[key] for key in before),
                  pair_both_wrong=sum(not before[key] and not after[key] for key in before),
                  exact_mcnemar_two_sided_p=p_value,
                  statistical_unit="paired before/after group")
    return result


def prepare_candidate(filename="candidate.py", function="prepare_ids"):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("llama_" + Path(filename).stem, HERE / filename)
    require(spec is not None and spec.loader is not None, f"Missing {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, function)


def audit_and_cost(cases, predictions, traces, strategy, preparers):
    expected_ids = {case["id"] for case in cases}
    selected_traces = [row for row in traces if row["strategy"] == strategy]
    require(len(selected_traces) == len(cases), f"Incomplete traces: {strategy}")
    indexed = {row["id"]: row for row in selected_traces}
    require(set(indexed) == expected_ids and len(indexed) == len(selected_traces),
            f"Duplicate or unknown trace identifiers: {strategy}")
    predicted = {row["id"]: row["response"] for row in predictions}
    issues, invalid, all_calls, walls, adjusted = [], [], [], [], []
    hashes_checked = 0
    for case in cases:
        trace = indexed[case["id"]]
        calls = trace["calls"]
        require(isinstance(calls, list) and bool(calls), "Trace has no calls")
        require(trace["original_messages_sha256"] == digest(case["messages"]),
                f"Original prompt hash mismatch: {case['id']}/{strategy}")
        _, relevant = gold_from_sources(case)
        if strategy == "direct":
            messages = [case["messages"]]
            require("extraction_response" not in trace, "Direct trace contains extraction")
            invalid_extraction = False
        else:
            prepared = preparers[strategy](case["messages"])
            messages = [prepared.extraction_messages]
            require(calls[0]["response"] == trace["extraction_response"],
                    "Raw extraction differs from saved request response")
            invalid_extraction = False
            try:
                if calls[0]["done_reason"] != "stop":
                    raise ValueError("Truncated extraction")
                final_messages = prepared.finish(trace["extraction_response"])
            except (ValueError, RecursionError):
                require(trace["extraction_valid"] is False, "Invalid extraction marked valid")
                require(predicted[case["id"]] == {"error": "invalid_extraction"},
                        "Invalid extraction did not produce the scored error marker")
                invalid.append(case["id"])
                invalid_extraction = True
            else:
                require(trace["extraction_valid"] is True, "Valid extraction marked invalid")
                messages.append(final_messages)
                _, selected = records_from(final_messages)
                actual = {record["source_id"] for record in selected}
                missing, extra = sorted(relevant - actual), sorted(actual - relevant)
                if missing or extra:
                    issues.append({"id": case["id"], "missing_source_ids": missing,
                                   "extra_source_ids": extra})
        require(len(calls) == len(messages), f"Wrong number of calls: {case['id']}/{strategy}")
        if not invalid_extraction:
            expected_response = (calls[-1]["response"] if calls[-1]["done_reason"] == "stop"
                                 else {"error": "truncated_response"})
            require(predicted[case["id"]] == expected_response,
                    "Prediction differs from raw final response or truncation marker")
        for call, sent in zip(calls, messages):
            require(call["messages_sha256"] == digest(sent),
                    f"Sent prompt hash mismatch: {case['id']}/{strategy}")
            require(call["done"] is True, "Incomplete saved model request")
            for field in ("eval_count", "prompt_eval_count", "load_duration", "total_duration"):
                require(type(call[field]) is int and call[field] >= 0,
                        f"Invalid request metric: {field}")
            require(isinstance(call["wall_seconds"], (float, int))
                    and math.isfinite(call["wall_seconds"]) and call["wall_seconds"] >= 0,
                    "Invalid per-request wall time")
            hashes_checked += 1
        wall = trace["wall_seconds"]
        require(isinstance(wall, (float, int)) and math.isfinite(wall) and wall >= 0,
                "Invalid per-case wall time")
        load_seconds = sum(call["load_duration"] for call in calls) / 1e9
        require(wall + 0.01 >= load_seconds, "Model loading exceeds measured case time")
        walls.append(wall)
        adjusted.append(max(0.0, wall - load_seconds))
        all_calls.extend(calls)
    reasons = defaultdict(int)
    for call in all_calls:
        reasons[str(call.get("done_reason"))] += 1
    return {
        "message_hashes_checked": hashes_checked,
        "cost": {
            "requests": len(all_calls), "total_wall_seconds": sum(walls),
            "median_case_wall_seconds": statistics.median(walls),
            "median_case_wall_seconds_minus_load": statistics.median(adjusted),
            "total_case_wall_seconds_minus_load": sum(adjusted),
            "output_tokens": sum(call["eval_count"] for call in all_calls),
            "prompt_tokens": sum(call["prompt_eval_count"] for call in all_calls),
            "max_output_tokens_per_call": max(call["eval_count"] for call in all_calls),
            "max_prompt_tokens_per_call": max(call["prompt_eval_count"] for call in all_calls),
            "done_reasons": dict(reasons),
            "non_stop_requests": sum(call.get("done_reason") != "stop" for call in all_calls),
            "thinking_responses": sum(bool(call.get("thinking_returned")) for call in all_calls),
        },
        "extraction": None if strategy == "direct" else {
            "invalid_ids": invalid,
            "valid_extractions": len(cases) - len(invalid),
            "exact_relevance_cases": len(cases) - len(invalid) - len(issues),
            "cases_with_missing_records": sum(bool(row["missing_source_ids"]) for row in issues),
            "cases_with_extra_records": sum(bool(row["extra_source_ids"]) for row in issues),
            "issues": issues,
        },
    }


def analyze(phase):
    strategies = (STRATEGIES if phase == "dev" else HOLDOUT_STRATEGIES
                  if phase == "holdout" else ("two-stage-ids-schema",))
    seed, groups = SETS[phase]
    cases = read_rows(HERE / f"{phase}-cases.jsonl")
    require(cases == generate_cases(seed=seed, groups=groups),
            f"{phase} cases differ from frozen seed {seed}, groups {groups}")
    ids = {case["id"] for case in cases}
    other_phase = "holdout" if phase.startswith("dev") else "dev"
    other_seed, other_groups = SETS[other_phase]
    other_ids = {case["id"] for case in generate_cases(seed=other_seed, groups=other_groups)}
    require(not ids & other_ids, "Development and holdout case IDs overlap")
    for case in cases:
        gold, _ = gold_from_sources(case)
        require(gold == case["expected"], f"Independent gold mismatch: {case['id']}")
    frozen_name = "frozen.json" if phase == "dev" else "schema-frozen.json"
    frozen = parse_json((HERE / frozen_name).read_text(encoding="utf-8"))
    if phase != "dev":
        parent_text = (HERE / "frozen.json").read_text(encoding="utf-8")
        require(hashlib.sha256(parent_text.encode("utf-8")).hexdigest()
                == frozen["parent_frozen_sha256"], "Parent freeze record changed after amendment")
    prompts = [{key: case[key] for key in ("id", "messages")} for case in cases]
    require(digest(prompts) == frozen["prompt_hashes"][phase], "Frozen prompt hash mismatch")
    require(frozen["phases"][phase] == {"seed": seed, "groups": groups, "cases": len(cases)},
            "Frozen phase settings differ from audited settings")
    predictions = {strategy: read_rows(HERE / f"{phase}-{strategy}.jsonl")
                   for strategy in strategies}
    reports = {strategy: score(cases, rows) for strategy, rows in predictions.items()}
    for strategy, rows in predictions.items():
        require(len(rows) == len(cases) and {row["id"] for row in rows} == ids,
                f"Incomplete predictions: {strategy}")
    traces = read_rows(HERE / f"{phase}-traces.jsonl")
    require(len(traces) == len(cases) * len(strategies)
            and all(row["strategy"] in strategies for row in traces),
            "Unknown, extra, or incomplete traces")
    orders = (list(itertools.permutations(strategies)) if phase == "dev" else
              [strategies[index:] + strategies[:index] for index in range(len(strategies))])
    expected_order = [(case["id"], strategy) for index, case in enumerate(cases)
                      for strategy in orders[index % len(orders)]]
    require([(row["id"], row["strategy"]) for row in traces] == expected_order,
            "Saved traces differ from the frozen balanced request order")
    require(all(call["model"] == frozen["model"]["name"] for row in traces for call in row["calls"]),
            "Model name differs from frozen model")
    preparers = {"two-stage": prepare, "two-stage-ids": prepare_candidate()}
    if "two-stage-ids-schema" in strategies:
        preparers["two-stage-ids-schema"] = prepare_candidate("schema_candidate.py", "prepare_schema_ids")
    result = {
        "phase": phase, "seed": seed, "groups": groups,
        "cases": len(cases), "pairs": len({case["pair_id"] for case in cases}),
        "languages": ["en", "de", "es"],
        "audit": {"independent_gold_checked": len(cases), "regenerated_cases_match": True,
                  "development_holdout_ids_disjoint": True, "balanced_trace_order_checked": True,
                  "frozen_prompt_hash_matches": True, "raw_responses_match_predictions": True},
        "strategies": {}, "comparisons": {},
        "interpretation": (
            "Development results are exploratory. The holdout contains new cases of the same "
            "three synthetic task types, not new task families. Pairs are the statistical unit. "
            "The ID strategy versus existing two-stage strategy is the primary improvement "
            "comparison; existing two-stage versus direct tests transfer. Holm adjustment "
            "covers those two planned comparisons. The direct versus ID comparison is exploratory. "
            "Two-stage strategies have up to twice the requests and output allowance of direct."
        ),
    }
    if phase != "dev":
        result["interpretation"] = (
            "The schema amendment was developed on the same 18 development cases and frozen "
            "before the 180-case holdout. Development results are exploratory and not new "
            "independent evidence. The holdout uses the same three synthetic task types, not "
            "new task families. Pairs are the statistical unit. ID selection plus an explicit "
            "final-output schema versus existing two-stage is the primary improvement comparison; "
            "existing two-stage versus direct tests transfer. Holm adjustment covers those two "
            "planned comparisons. All other comparisons, including plain IDs versus IDs plus "
            "schema, are exploratory ablations. Two-stage strategies allow up to twice the "
            "requests and output allowance of direct."
        )
    for strategy, report in reports.items():
        result["strategies"][strategy] = {
            key: report[key] for key in ("summary", "by_task", "by_language")}
        result["strategies"][strategy]["strict_correct"] = sum(row["correct"] for row in report["results"])
        result["strategies"][strategy]["pairs_correct"] = sum(paired(report).values())
        result["strategies"][strategy].update(
            audit_and_cost(cases, predictions[strategy], traces, strategy, preparers))
    for old, new in itertools.combinations(strategies, 2):
        result["comparisons"][old + "_vs_" + new] = paired_comparison(
            cases, predictions, reports, old, new)
    if phase == "dev-schema":
        result["integration_gate"] = {
            "frozen_rule": frozen["integration_gate"], "decision_eligible": False, "passes": False,
            "note": "Reused development cases; no paired comparison or adoption decision is made.",
        }
        return result
    candidate_strategy = "two-stage-ids" if phase == "dev" else "two-stage-ids-schema"
    planned = ["direct_vs_two-stage", "two-stage_vs_" + candidate_strategy]
    ranked = sorted(planned, key=lambda key: result["comparisons"][key]["exact_mcnemar_two_sided_p"])
    previous = 0.0
    for index, key in enumerate(ranked):
        comparison = result["comparisons"][key]
        previous = max(previous, min(1.0, comparison["exact_mcnemar_two_sided_p"] * (len(ranked) - index)))
        comparison["holm_adjusted_p_two_planned_comparisons"] = previous
    old, new = (result["strategies"][name] for name in ("two-stage", candidate_strategy))
    tokens_reduction = (1 - new["cost"]["output_tokens"] / old["cost"]["output_tokens"]
                        if old["cost"]["output_tokens"] else None)
    median_reduction = (1 - new["cost"]["median_case_wall_seconds"] / old["cost"]["median_case_wall_seconds"]
                        if old["cost"]["median_case_wall_seconds"] else None)
    quality_preserved = (new["pairs_correct"] >= old["pairs_correct"]
                         and new["strict_correct"] >= old["strict_correct"])
    quality_route = (new["pairs_correct"] > old["pairs_correct"]
                     and tokens_reduction is not None and tokens_reduction > 0)
    efficiency_route = (tokens_reduction is not None and median_reduction is not None
                        and tokens_reduction >= 0.25 and median_reduction >= 0.15)
    result["integration_gate"] = {
        "frozen_rule": frozen["integration_gate"],
        "decision_eligible": phase == "holdout",
        "quality_non_decreasing": quality_preserved,
        "output_token_reduction_fraction": tokens_reduction,
        "median_wall_reduction_fraction": median_reduction,
        "quality_improvement_route": quality_route,
        "efficiency_route": efficiency_route,
        "passes": phase == "holdout" and quality_preserved and (quality_route or efficiency_route),
        "note": "Development is excluded from adoption decisions. Aggregate non-regression does not imply zero individually regressed cases.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", nargs="?", default="dev", choices=SETS)
    parser.add_argument("--out", type=Path, help="New output path; never overwritten")
    args = parser.parse_args()
    path = args.out if args.out is not None else HERE / f"{args.phase}-analysis.json"
    if path.exists():
        parser.error(f"Output already exists: {path}")
    result = analyze(args.phase)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    for strategy, report in result["strategies"].items():
        print(f"{strategy}: {report['strict_correct']}/{result['cases']} cases; "
              f"{report['pairs_correct']}/{result['pairs']} pairs; "
              f"{report['cost']['requests']} requests")
    print(f"Verified independent gold and every model-message hash; saved {path}")


if __name__ == "__main__":
    main()
