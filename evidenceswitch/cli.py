"""Command-line interface. Core generation and scoring run entirely offline."""

import argparse
import json
import math
import sys
from pathlib import Path

from .generate import generate_cases
from .io import read_jsonl, write_json, write_jsonl
from .score import compare, score


def _dataset(out, seed, groups, languages):
    cases = generate_cases(seed=seed, groups=groups, languages=languages)
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "cases.jsonl", cases)
    write_jsonl(out / "prompts.jsonl", ({"id": c["id"], "messages": c["messages"]} for c in cases))
    write_json(out / "manifest.json", {"schema_version": 1, "generator_version": "0.1.0",
                                      "seed": seed, "groups_per_task_language": groups,
                                      "languages": list(languages), "cases": len(cases)})
    return cases


def _parser():
    parser = argparse.ArgumentParser(description="EvidenceSwitch: synthetic evidence regression checks")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("generate", "Create cases and separate model prompts"),
                            ("demo", "Run simulated responders offline (not a model evaluation)")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--out", type=Path, required=True, help="New output directory")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--groups", type=int, default=10, help="Pairs per task per language (default 10)")
        command.add_argument("--languages", nargs="+", choices=("en", "de", "es"), default=["en", "de", "es"])
    evaluate = commands.add_parser("score", help="Score responses against cases")
    evaluate.add_argument("--cases", required=True)
    evaluate.add_argument("--responses", required=True)
    evaluate.add_argument("--out", type=Path)
    evaluate.add_argument("--min-pair-accuracy", type=float, default=0.0)
    comparison = commands.add_parser("compare", help="Find fixed and regressed cases")
    comparison.add_argument("--cases", required=True)
    comparison.add_argument("--baseline", required=True)
    comparison.add_argument("--candidate", required=True)
    comparison.add_argument("--out", type=Path)
    comparison.add_argument("--fail-on-regression", action="store_true")
    run = commands.add_parser("run", help="Explicitly send prompts to a model endpoint (may incur costs)")
    run.add_argument("--prompts", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--base-url", required=True, help="Endpoint API prefix, e.g. http://localhost:8000/v1")
    run.add_argument("--model", required=True)
    run.add_argument("--api-key-env", help="Name of environment variable containing endpoint API key")
    run.add_argument("--limit", type=int)
    run.add_argument("--timeout", type=float, default=60.0)
    run.add_argument("--strategy", choices=("direct", "two-stage", "two-stage-ids"), default="direct")
    run.add_argument("--trace", help="New JSONL file for extraction responses and request metadata")
    local = commands.add_parser("run-ollama", help="Use an installed local Ollama model (no paid API or downloads)")
    local.add_argument("--prompts", required=True)
    local.add_argument("--out", required=True)
    local.add_argument("--model", required=True)
    local.add_argument("--base-url", default="http://127.0.0.1:11434", help="Local loopback origin only")
    local.add_argument("--strategy", choices=("direct", "two-stage", "two-stage-ids"), default="direct")
    local.add_argument("--trace", help="New JSONL file for extraction responses, timing, tokens and settings")
    local.add_argument("--limit", type=int)
    local.add_argument("--timeout", type=float, default=180.0)
    local.add_argument("--seed", type=int, default=20261015)
    local.add_argument("--num-ctx", type=int, default=4096)
    local.add_argument("--num-predict", type=int, default=512, help="Maximum output tokens per model call")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            cases = _dataset(args.out, args.seed, args.groups, args.languages)
            print(f"Created {len(cases)} cases in {args.out}. Only send prompts.jsonl to models.")
        elif args.command == "demo":
            cases = _dataset(args.out, args.seed, args.groups, args.languages)
            originals = {c["pair_id"]: c["expected"] for c in cases if c["variant"] == "before"}
            oracle = [{"id": c["id"], "response": c["expected"]} for c in cases]
            stale = [{"id": c["id"], "response": originals[c["pair_id"]]} for c in cases]
            for name, responses in (("oracle", oracle), ("stale", stale)):
                write_jsonl(args.out / f"simulated-{name}.jsonl", responses)
                report = score(cases, responses)
                report["simulation"] = name
                write_json(args.out / f"simulated-{name}-report.json", report)
                print(f"SIMULATED {name}: {report['summary']['pair_accuracy']:.1%} pair accuracy")
            print("These fixtures demonstrate scoring, not real model performance.")
        elif args.command == "run":
            from .runner import run_prompts
            count = run_prompts(args.prompts, args.out, base_url=args.base_url, model=args.model,
                                api_key_env=args.api_key_env, limit=args.limit, timeout=args.timeout,
                                strategy=args.strategy, trace_path=args.trace)
            print(f"Saved {count} responses to {args.out}")
        elif args.command == "run-ollama":
            from .ollama import run_ollama
            count = run_ollama(args.prompts, args.out, base_url=args.base_url, model=args.model,
                               strategy=args.strategy, trace_path=args.trace, limit=args.limit,
                               timeout=args.timeout, seed=args.seed, num_ctx=args.num_ctx,
                               num_predict=args.num_predict)
            print(f"Saved {count} responses to {args.out}")
        else:
            if args.command == "score":
                if not math.isfinite(args.min_pair_accuracy) or not 0 <= args.min_pair_accuracy <= 1:
                    raise ValueError("--min-pair-accuracy must be between 0 and 1")
                report = score(read_jsonl(args.cases), read_jsonl(args.responses, allow_empty=True))
                failed = report["summary"]["pair_accuracy"] < args.min_pair_accuracy
            else:
                report = compare(read_jsonl(args.cases), read_jsonl(args.baseline, allow_empty=True),
                                 read_jsonl(args.candidate, allow_empty=True))
                failed = args.fail_on_regression and bool(report["regressed_ids"])
            if args.out:
                write_json(args.out, report)
            printable = report["summary"] if args.command == "score" else report
            print(json.dumps(printable, indent=2, allow_nan=False))
            if failed:
                return 1
    except (ValueError, OSError, RuntimeError, RecursionError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0
