import copy
import unittest

from evidenceswitch.generate import generate_cases
from evidenceswitch.score import compare, score


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.cases = generate_cases(groups=1)
        self.oracle = [{"id": c["id"], "response": copy.deepcopy(c["expected"])} for c in self.cases]

    def test_oracle_and_stale_answer_baselines(self):
        perfect = score(self.cases, self.oracle)
        self.assertEqual(perfect["summary"]["pair_accuracy"], 1)
        originals = {c["pair_id"]: c["expected"] for c in self.cases if c["variant"] == "before"}
        stale = [{"id": c["id"], "response": originals[c["pair_id"]]} for c in self.cases]
        report = score(self.cases, stale)
        self.assertAlmostEqual(report["summary"]["pair_accuracy"], 1 / 3)
        self.assertAlmostEqual(report["summary"]["strict_accuracy"], 2 / 3)
        self.assertEqual(report["by_task"]["distractor"]["pair_accuracy"], 1)
        self.assertEqual(report["by_task"]["update"]["pair_accuracy"], 0)
        self.assertEqual(report["by_task"]["missing"]["pair_accuracy"], 0)

    def test_empty_and_partial_predictions_never_drop_missing_cases(self):
        empty = score(self.cases, [])
        self.assertEqual(empty["summary"]["coverage"], 0)
        self.assertEqual(empty["summary"]["pair_accuracy"], 0)
        partial = score(self.cases, self.oracle[:1])
        self.assertEqual(partial["summary"]["cases"], 18)
        self.assertEqual(partial["summary"]["strict_accuracy"], 1 / 18)
        self.assertEqual(partial["summary"]["pair_accuracy"], 0)

    def test_answer_and_citations_are_scored_separately(self):
        index = next(i for i, c in enumerate(self.cases) if c["expected"]["answer"] is not None)
        self.oracle[index]["response"]["citations"] = ["wrong-source"]
        result = score(self.cases, self.oracle)["results"][index]
        self.assertTrue(result["format_valid"])
        self.assertTrue(result["answer_correct"])
        self.assertFalse(result["citations_correct"])
        self.assertFalse(result["correct"])

    def test_invalid_model_output_fails_without_crashing(self):
        malformed = [
            '```json\n{"answer":null,"citations":[]}\n```',
            '{"answer":null,"answer":null,"citations":[]}',
            '{"answer":NaN,"citations":[]}',
            {"answer": None, "citations": ["x", "x"]},
            {"answer": None, "citations": [], "explanation": "extra"},
            {"answer": 4, "citations": []},
            {"answer": None, "citations": [[]]},
            None, [], "not json",
        ]
        for response in malformed:
            with self.subTest(response=response):
                output = [{"id": self.cases[0]["id"], "response": response}]
                result = score(self.cases, output)["results"][0]
                self.assertFalse(result["format_valid"])
                self.assertFalse(result["correct"])

    def test_always_abstaining_does_not_pass_pairs(self):
        responses = [{"id": c["id"], "response": {"answer": None, "citations": []}} for c in self.cases]
        summary = score(self.cases, responses)["summary"]
        self.assertEqual(summary["pair_accuracy"], 0)
        self.assertEqual(summary["abstention_accuracy"], 1)
        self.assertEqual(summary["false_abstention_rate"], 1)
        self.assertAlmostEqual(summary["strict_accuracy"], 1 / 6)

    def test_bad_prediction_identifiers_are_not_silently_ignored(self):
        for predictions in ([self.oracle[0], self.oracle[0]],
                            [{"id": "unknown", "response": {}}],
                            [{"id": [], "response": {}}]):
            with self.assertRaises(ValueError):
                score(self.cases, predictions)

    def test_broken_pairs_and_gold_are_rejected(self):
        with self.assertRaises(ValueError):
            score(self.cases[:-1], self.oracle)
        broken = copy.deepcopy(self.cases)
        broken[0]["expected"] = {"answer": None, "citations": ["x"]}
        with self.assertRaises(ValueError):
            score(broken, [])

    def test_compare_exposes_regression_even_when_net_accuracy_is_unchanged(self):
        baseline, candidate = copy.deepcopy(self.oracle), copy.deepcopy(self.oracle)
        baseline[0]["response"] = "broken"
        candidate[1]["response"] = "broken"
        result = compare(self.cases, baseline, candidate)
        self.assertEqual(result["baseline"]["strict_accuracy"], result["candidate"]["strict_accuracy"])
        self.assertEqual(result["fixed_ids"], [self.cases[0]["id"]])
        self.assertEqual(result["regressed_ids"], [self.cases[1]["id"]])


if __name__ == "__main__":
    unittest.main()
