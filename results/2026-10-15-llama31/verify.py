"""Verify complete published Llama and Qwen evidence offline, without a model.

Run this file from any directory. Missing or inconsistent artifacts fail the
audit. No files are written, no model is called, and no network is used.
"""

import importlib.util
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
spec = importlib.util.spec_from_file_location("llama_evidence_analysis", HERE / "analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

from evidenceswitch.selection import prepare_ids
from evidenceswitch.workflow import prepare

require = analysis.require
digest = analysis.digest
read_rows = analysis.read_rows


def read_json(name):
    return analysis.parse_json((HERE / name).read_text(encoding="utf-8"))


def same_report(actual, saved, path="report"):
    """Exact structure/counts; tolerate only interpreter float-sum rounding."""
    require(type(actual) is type(saved), f"Changed type: {path}")
    if isinstance(actual, dict):
        require(set(actual) == set(saved), f"Changed keys: {path}")
        for key in actual:
            same_report(actual[key], saved[key], f"{path}.{key}")
    elif isinstance(actual, list):
        require(len(actual) == len(saved), f"Changed list length: {path}")
        for index, (left, right) in enumerate(zip(actual, saved)):
            same_report(left, right, f"{path}[{index}]")
    elif isinstance(actual, float):
        require(math.isclose(actual, saved, rel_tol=1e-12, abs_tol=0), f"Changed number: {path}")
    else:
        require(actual == saved, f"Changed value: {path}")


def production_replay():
    """Check that integrated ID mode sends the messages actually measured."""
    cases = {case["id"]: case for case in read_rows(HERE / "holdout-cases.jsonl")}
    traces = [row for row in read_rows(HERE / "holdout-traces.jsonl")
              if row["strategy"] == "two-stage-ids-schema"]
    require(len(traces) == len(cases) and {row["id"] for row in traces} == set(cases),
            "Incomplete production replay cases")
    checked = 0
    for trace in traces:
        prepared = prepare_ids(cases[trace["id"]]["messages"])
        messages = [prepared.extraction_messages]
        try:
            if trace["calls"][0]["done_reason"] != "stop":
                raise ValueError("Truncated extraction")
            final_messages = prepared.finish(trace["extraction_response"])
        except (ValueError, RecursionError):
            require(trace["extraction_valid"] is False, "Production rejects a measured valid extraction")
        else:
            require(trace["extraction_valid"] is True, "Production accepts a measured invalid extraction")
            messages.append(final_messages)
        require(len(messages) == len(trace["calls"]), "Production call count changed")
        for sent, call in zip(messages, trace["calls"]):
            require(digest(sent) == call["messages_sha256"], "Production prompt differs from experiment")
            checked += 1
    return {"cases": len(traces), "message_hashes_checked": checked}


def qwen_analysis():
    """Recompute the separate, conditional 36-case compatibility check.

    The native runner stores raw final answers in predictions, and extraction
    text plus message hashes in traces. Any non-stop request fails this audit;
    it is never silently repaired or counted as a completed normal response.
    """
    plan = read_json("qwen-regression-plan.json")
    require((plan["seed"], plan["groups"], plan["cases"]) == (909, 2, 36), "Qwen plan size changed")
    require(plan["languages"] == ["en", "de", "es"] and plan["think"] is False,
            "Qwen language or thinking settings changed")
    require(plan["strategies"] == ["two-stage", "two-stage-ids"], "Qwen strategies changed")
    cases = read_rows(HERE / "qwen-cases.jsonl")
    require(cases == analysis.generate_cases(seed=909, groups=2), "Qwen generated cases differ")
    prompts = [{key: case[key] for key in ("id", "messages")} for case in cases]
    require(read_rows(HERE / "qwen-prompts.jsonl") == prompts, "Qwen prompts contain changed data")
    require(digest(prompts) == plan["prompt_hash"], "Qwen frozen prompt hash differs")
    require(read_json("holdout-analysis.json")["integration_gate"]["passes"] is True,
            "Conditional Qwen run requires the frozen Llama integration gate to pass")
    ids = {case["id"] for case in cases}
    previous_ids = {case["id"] for phase in ("dev", "holdout")
                    for case in read_rows(HERE / f"{phase}-cases.jsonl")}
    require(not ids & previous_ids, "Qwen cases overlap Llama development or holdout")
    for case in cases:
        require(analysis.gold_from_sources(case)[0] == case["expected"], "Qwen independent gold differs")
    reports, predictions, strategies = {}, {}, {}
    environment = None
    for strategy in plan["strategies"]:
        predictions[strategy] = read_rows(HERE / f"qwen-{strategy}.jsonl")
        reports[strategy] = analysis.score(cases, predictions[strategy])
        require(len(predictions[strategy]) == len(cases)
                and {row["id"] for row in predictions[strategy]} == ids,
                "Incomplete Qwen predictions")
        by_id = {row["id"]: row["response"] for row in predictions[strategy]}
        traces = read_rows(HERE / f"qwen-{strategy}-trace.jsonl")
        require([row["id"] for row in traces] == [case["id"] for case in cases],
                "Incomplete or reordered Qwen traces")
        walls, adjusted, calls, invalid, issues = [], [], [], [], []
        hashes_checked = 0
        for case, trace in zip(cases, traces):
            require(trace["strategy"] == strategy, "Qwen trace strategy differs")
            require(trace["original_messages_sha256"] == digest(case["messages"]),
                    "Qwen original prompt hash differs")
            current = trace["environment"]
            require(current["provider"] == "ollama" and current["model"] == plan["model"]
                    and current["digest"] == plan["model_digest"]
                    and current["options"] == plan["options"] and current["think"] is False,
                    "Qwen model digest or inference settings differ from plan")
            if environment is None:
                environment = current
            require(current == environment, "Qwen environment changed between requests")
            recorded = trace["calls"]
            require(recorded and all(call["done"] is True and call["done_reason"] == "stop"
                                     for call in recorded), "Qwen contains a non-stop or incomplete request")
            prepared = (prepare if strategy == "two-stage" else prepare_ids)(case["messages"])
            messages = [prepared.extraction_messages]
            try:
                final_messages = prepared.finish(trace["extraction_response"])
            except (ValueError, RecursionError):
                require(trace["extraction_valid"] is False
                        and by_id[case["id"]] == {"error": "invalid_extraction"},
                        "Invalid Qwen extraction was not counted as a failure")
                invalid.append(case["id"])
            else:
                require(trace["extraction_valid"] is True, "Qwen extraction flag is incorrect")
                require(isinstance(by_id[case["id"]], str), "Qwen final response is not raw text")
                messages.append(final_messages)
                _, selected = analysis.records_from(final_messages)
                relevant = analysis.gold_from_sources(case)[1]
                selected_ids = {record["source_id"] for record in selected}
                missing, extra = sorted(relevant - selected_ids), sorted(selected_ids - relevant)
                if missing or extra:
                    issues.append({"id": case["id"], "missing_source_ids": missing,
                                   "extra_source_ids": extra})
            require(len(recorded) == len(messages), "Qwen retry, fallback, or missing call detected")
            for sent, call in zip(messages, recorded):
                require(digest(sent) == call["messages_sha256"], "Qwen sent prompt hash differs")
                require(call["model"] == plan["model"] and call["thinking_returned"] is False,
                        "Qwen model or thinking output differs")
                for key in ("eval_count", "prompt_eval_count", "load_duration", "total_duration"):
                    require(type(call[key]) is int and call[key] >= 0, "Invalid Qwen request metric")
                require(isinstance(call["wall_seconds"], (int, float))
                        and math.isfinite(call["wall_seconds"]) and call["wall_seconds"] >= 0,
                        "Invalid Qwen request wall time")
                hashes_checked += 1
            wall = trace["wall_seconds"]
            load = sum(call["load_duration"] for call in recorded) / 1e9
            require(isinstance(wall, (int, float)) and math.isfinite(wall) and wall >= 0
                    and wall + 0.01 >= load, "Invalid Qwen case wall time")
            walls.append(wall)
            adjusted.append(max(0.0, wall - load))
            calls.extend(recorded)
        report = reports[strategy]
        strategies[strategy] = {key: report[key] for key in ("summary", "by_task", "by_language")}
        strategies[strategy].update(
            strict_correct=sum(row["correct"] for row in report["results"]),
            pairs_correct=sum(analysis.paired(report).values()),
            message_hashes_checked=hashes_checked,
            environment=environment,
            cost={"requests": len(calls), "total_wall_seconds": sum(walls),
                  "median_case_wall_seconds": statistics.median(walls),
                  "median_case_wall_seconds_minus_load": statistics.median(adjusted),
                  "output_tokens": sum(call["eval_count"] for call in calls),
                  "prompt_tokens": sum(call["prompt_eval_count"] for call in calls),
                  "max_output_tokens_per_call": max(call["eval_count"] for call in calls),
                  "max_prompt_tokens_per_call": max(call["prompt_eval_count"] for call in calls),
                  "non_stop_requests": 0, "thinking_responses": 0},
            extraction={"invalid_ids": invalid, "valid_extractions": len(cases) - len(invalid),
                        "exact_relevance_cases": len(cases) - len(invalid) - len(issues),
                        "cases_with_missing_records": sum(bool(row["missing_source_ids"]) for row in issues),
                        "cases_with_extra_records": sum(bool(row["extra_source_ids"]) for row in issues),
                        "issues": issues})
    return {
        "phase": "qwen-compatibility", "seed": 909, "groups": 2, "cases": 36, "pairs": 18,
        "strategies": strategies,
        "comparison": analysis.compare(cases, predictions["two-stage"], predictions["two-stage-ids"]),
        "audit": {"independent_gold_checked": len(cases), "regenerated_cases_match": True,
                  "frozen_prompt_hash_matches": True, "case_ids_disjoint_from_llama": True,
                  "environment_matches_plan": True, "all_requests_stopped_normally": True},
        "interpretation": "Conditional compatibility check on the original model; excluded from the Llama primary result. Sequential conditions make timing descriptive. Production two-stage-ids is the experimental ID-plus-schema method, not the plain-ID ablation.",
    }


def main():
    for phase in ("dev", "dev-schema", "holdout"):
        same_report(analysis.analyze(phase), read_json(f"{phase}-analysis.json"), phase)
        print(f"Verified saved {phase} results and original request hashes.")
    replay = production_replay()
    print(f"Production matches {replay['message_hashes_checked']} measured request hashes "
          f"across {replay['cases']} Llama cases.")
    import hashlib
    from content_diagnostic import analyze_content
    plan = read_json("content-diagnostic-plan.json")
    source = (HERE / "content_diagnostic.py").read_text(encoding="utf-8").encode("utf-8")
    require(hashlib.sha256(source).hexdigest() == plan["script_sha256"], "Diagnostic code changed")
    actual = analyze_content("holdout")
    saved = read_json("holdout-content-diagnostic.json")
    actual.pop("created_utc")
    saved.pop("created_utc")
    same_report(actual, saved, "content-diagnostic")
    print("Verified supplementary citation-format diagnostic without changing strict scores.")
    same_report(qwen_analysis(), read_json("qwen-analysis.json"), "qwen")
    print("Verified 36 conditional Qwen cases, both strategies, and all saved metrics. No model calls.")


if __name__ == "__main__":
    main()
