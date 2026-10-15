"""Deterministic scoring, without a model-based judge."""

from collections import defaultdict

from .io import parse_json


def valid_answer(value):
    return (
        isinstance(value, dict)
        and set(value) == {"answer", "citations"}
        and (value["answer"] is None or isinstance(value["answer"], str))
        and isinstance(value["citations"], list)
        and all(isinstance(item, str) and item for item in value["citations"])
        and len(value["citations"]) == len(set(value["citations"]))
    )


def validate_cases(cases):
    if not cases:
        raise ValueError("Cases must not be empty")
    seen, pairs = set(), defaultdict(list)
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object")
        for key in ("id", "pair_id", "task", "language", "variant"):
            if not isinstance(case.get(key), str) or not case[key]:
                raise ValueError(f"Case requires a nonempty {key}")
        if case["id"] in seen:
            raise ValueError(f"Duplicate case ID: {case['id']}")
        seen.add(case["id"])
        if case["task"] not in {"update", "missing", "distractor"}:
            raise ValueError("Unknown task")
        if case["language"] not in {"en", "de", "es"}:
            raise ValueError("Unknown language")
        if not valid_answer(case.get("expected")):
            raise ValueError("Invalid expected answer")
        gold = case["expected"]
        if (gold["answer"] is None) != (gold["citations"] == []):
            raise ValueError("Gold abstentions require no citations; answers require citations")
        pairs[case["pair_id"]].append(case)
    for pair in pairs.values():
        if len(pair) != 2 or {c["variant"] for c in pair} != {"before", "after"}:
            raise ValueError("Each pair requires exactly one before and one after case")
        if len({(c["task"], c["language"]) for c in pair}) != 1:
            raise ValueError("Pair task and language must agree")


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _summary(results):
    pairs = defaultdict(list)
    for result in results:
        pairs[result["pair_id"]].append(result)
    absent = [r for r in results if r["expected_abstention"]]
    answerable = [r for r in results if not r["expected_abstention"]]
    return {
        "cases": len(results),
        "pairs": len(pairs),
        "responses": sum(r["present"] for r in results),
        "coverage": _rate(sum(r["present"] for r in results), len(results)),
        "format_accuracy": _rate(sum(r["format_valid"] for r in results), len(results)),
        "answer_accuracy": _rate(sum(r["answer_correct"] for r in results), len(results)),
        "citation_accuracy": _rate(sum(r["citations_correct"] for r in results), len(results)),
        "strict_accuracy": _rate(sum(r["correct"] for r in results), len(results)),
        "pair_accuracy": _rate(sum(all(r["correct"] for r in p) for p in pairs.values()), len(pairs)),
        "abstention_accuracy": _rate(sum(r["abstained"] for r in absent), len(absent)),
        "false_abstention_rate": _rate(sum(r["abstained"] for r in answerable), len(answerable)),
    }


def score(cases, predictions):
    """Missing responses fail; duplicate/unknown IDs are input errors."""
    validate_cases(cases)
    case_ids = {case["id"] for case in cases}
    indexed = {}
    for row in predictions:
        if not isinstance(row, dict) or set(row) != {"id", "response"}:
            raise ValueError("Each prediction must contain only id and response")
        case_id = row["id"]
        if not isinstance(case_id, str) or case_id not in case_ids:
            raise ValueError("Prediction has an unknown case ID")
        if case_id in indexed:
            raise ValueError(f"Duplicate prediction ID: {case_id}")
        indexed[case_id] = row["response"]
    results = []
    for case in cases:
        raw = indexed.get(case["id"])
        try:
            parsed = parse_json(raw) if isinstance(raw, str) else raw
        except (ValueError, RecursionError):
            parsed = None
        valid = valid_answer(parsed)
        answer_ok = valid and parsed["answer"] == case["expected"]["answer"]
        citations_ok = valid and set(parsed["citations"]) == set(case["expected"]["citations"])
        results.append({
            "id": case["id"], "pair_id": case["pair_id"],
            "task": case["task"], "language": case["language"],
            "present": case["id"] in indexed, "format_valid": valid,
            "answer_correct": answer_ok, "citations_correct": citations_ok,
            "correct": answer_ok and citations_ok,
            "expected_abstention": case["expected"]["answer"] is None,
            "abstained": valid and parsed["answer"] is None,
        })
    return {
        "schema_version": 1,
        "summary": _summary(results),
        "by_task": {key: _summary([r for r in results if r["task"] == key])
                    for key in sorted({r["task"] for r in results})},
        "by_language": {key: _summary([r for r in results if r["language"] == key])
                        for key in sorted({r["language"] for r in results})},
        "results": results,
    }


def compare(cases, baseline, candidate):
    """Compare the same fixtures without letting improvements hide regressions."""
    old, new = score(cases, baseline), score(cases, candidate)
    fixes, regressions = [], []
    for before, after in zip(old["results"], new["results"]):
        if before["correct"] and not after["correct"]:
            regressions.append(before["id"])
        if not before["correct"] and after["correct"]:
            fixes.append(before["id"])
    return {"schema_version": 1, "baseline": old["summary"],
            "candidate": new["summary"], "fixed_ids": fixes,
            "regressed_ids": regressions}
