import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evidenceswitch.cli import main
from evidenceswitch.io import read_jsonl


class CliTests(unittest.TestCase):
    def invoke(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(list(map(str, args)))

    def test_offline_demo_scoring_and_regression_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "demo"
            self.assertEqual(self.invoke("demo", "--out", out, "--groups", 1), 0)
            cases = out / "cases.jsonl"
            oracle = out / "simulated-oracle.jsonl"
            stale = out / "simulated-stale.jsonl"
            self.assertEqual(self.invoke("score", "--cases", cases, "--responses", oracle,
                                         "--min-pair-accuracy", 1), 0)
            self.assertEqual(self.invoke("score", "--cases", cases, "--responses", stale,
                                         "--min-pair-accuracy", 1), 1)
            self.assertEqual(self.invoke("compare", "--cases", cases, "--baseline", oracle,
                                         "--candidate", stale, "--fail-on-regression"), 1)
            self.assertEqual(self.invoke("compare", "--cases", cases, "--baseline", stale,
                                         "--candidate", oracle, "--fail-on-regression"), 0)
            prompts = read_jsonl(out / "prompts.jsonl")
            self.assertEqual(len(prompts), 18)
            self.assertTrue(all(set(p) == {"id", "messages"} for p in prompts))

    def test_empty_prediction_file_has_zero_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "data"
            empty = Path(directory) / "empty.jsonl"
            empty.touch()
            self.assertEqual(self.invoke("generate", "--out", out, "--groups", 1), 0)
            self.assertEqual(self.invoke("score", "--cases", out / "cases.jsonl", "--responses", empty,
                                         "--min-pair-accuracy", 1), 1)

    def test_output_is_not_overwritten_and_invalid_threshold_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "data"
            self.assertEqual(self.invoke("generate", "--out", out, "--groups", 1), 0)
            original = (out / "cases.jsonl").read_bytes()
            self.assertEqual(self.invoke("generate", "--out", out, "--groups", 1), 2)
            self.assertEqual((out / "cases.jsonl").read_bytes(), original)
            self.assertEqual(self.invoke("score", "--cases", out / "cases.jsonl", "--responses", "unused",
                                         "--min-pair-accuracy", "nan"), 2)

    def test_two_stage_runner_options_reach_both_transports(self):
        with patch("evidenceswitch.ollama.run_ollama", return_value=6) as local:
            self.assertEqual(self.invoke("run-ollama", "--prompts", "prompts.jsonl", "--out", "out.jsonl",
                                         "--model", "local-model", "--strategy", "two-stage", "--trace", "trace.jsonl"), 0)
        self.assertEqual(local.call_args.kwargs["strategy"], "two-stage")
        self.assertEqual(local.call_args.kwargs["trace_path"], "trace.jsonl")
        self.assertEqual(local.call_args.kwargs["num_predict"], 512)
        with patch("evidenceswitch.runner.run_prompts", return_value=6) as compatible:
            self.assertEqual(self.invoke("run", "--prompts", "prompts.jsonl", "--out", "out.jsonl",
                                         "--model", "model", "--base-url", "http://localhost:8000/v1",
                                         "--strategy", "two-stage", "--trace", "trace.jsonl"), 0)
        self.assertEqual(compatible.call_args.kwargs["strategy"], "two-stage")
        self.assertEqual(compatible.call_args.kwargs["trace_path"], "trace.jsonl")

    def test_selection_and_retrieval_are_available_for_both_transports(self):
        for strategy in ("two-stage-ids", "two-stage-retrieval", "two-stage-latest"):
            for command, target in (("run", "evidenceswitch.runner.run_prompts"),
                                    ("run-ollama", "evidenceswitch.ollama.run_ollama")):
                with self.subTest(command=command, strategy=strategy), patch(target, return_value=1) as run:
                    self.assertEqual(self.invoke(command, "--prompts", "prompts.jsonl", "--out", "out.jsonl",
                                                 "--model", "model", "--base-url", "http://localhost:8000",
                                                 "--strategy", strategy), 0)
                    self.assertEqual(run.call_args.kwargs["strategy"], strategy)


if __name__ == "__main__":
    unittest.main()
