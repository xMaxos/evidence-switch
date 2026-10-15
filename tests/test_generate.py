import collections
import json
import random
import re
import unittest
from unittest.mock import patch

from evidenceswitch.generate import _LANGUAGES, generate_cases


def records(case):
    return json.loads(case["messages"][1]["content"].split("\n\n", 1)[1].split("\n", 1)[1])


def pairs(cases):
    grouped = collections.defaultdict(dict)
    for case in cases:
        grouped[case["pair_id"]][case["variant"]] = case
    return list(grouped.values())


def question_target(case):
    question = case["messages"][1]["content"].split("\n", 1)[0]
    entity = re.search(r"Nuvora-[0-9a-f]{12}", question).group()
    field = re.search(r"signal_code|route_token|archive_label|beacon_key", question).group()
    return entity, field


class GenerateTests(unittest.TestCase):
    def test_default_counts_unique_ids_pairs_and_no_label_leaks(self):
        cases = generate_cases()
        self.assertEqual(len(cases), 180)
        self.assertEqual(len({case["id"] for case in cases}), 180)
        self.assertEqual(len(pairs(cases)), 90)
        self.assertEqual(collections.Counter((c["task"], c["language"]) for c in cases),
                         {(t, lang): 20 for t in ("update", "missing", "distractor")
                          for lang in ("en", "de", "es")})
        for pair in pairs(cases):
            self.assertEqual(set(pair), {"before", "after"})
        for case in cases:
            self.assertRegex(case["id"], r"^[0-9a-f]{12}$")
            self.assertRegex(case["pair_id"], r"^[0-9a-f]{12}$")
            self.assertEqual([m["role"] for m in case["messages"]], ["system", "user"])
            message_text = json.dumps(case["messages"])
            for secret in (case["id"], case["pair_id"], '"expected"', '"variant"'):
                self.assertNotIn(secret, message_text)
            self.assertEqual(len(records(case)), 8)
            self.assertEqual(len({r["source_id"] for r in records(case)}), 8)
            for record in records(case):
                self.assertRegex(record["source_id"], r"^[0-9a-f]{12}$")
            if case["variant"] == "before":
                combinations = collections.Counter((r["entity"], r["field"]) for r in records(case))
                self.assertEqual(sorted(combinations.values()), [2, 2, 2, 2])

    def test_seed_reproducibility_difference_and_global_rng_unchanged(self):
        state = random.getstate()
        first = generate_cases(seed=123, groups=2)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(first, generate_cases(seed=123, groups=2))
        self.assertNotEqual(first, generate_cases(seed=124, groups=2))
        self.assertEqual(random.getstate(), state)

    def test_gold_is_exact_highest_numeric_matching_source(self):
        lexical_disagreements = 0
        winning_revisions = set()
        for case in generate_cases(groups=4):
            entity, field = question_target(case)
            matching = [r for r in records(case) if (r["entity"], r["field"]) == (entity, field)]
            if matching:
                winner = max(matching, key=lambda r: r["revision"])
                winning_revisions.add(winner["revision"])
                self.assertEqual(case["expected"], {
                    "answer": winner["value"], "citations": [winner["source_id"]]})
                if len(matching) > 1:
                    lexical = max(matching, key=lambda r: str(r["revision"]))
                    lexical_disagreements += winner["source_id"] != lexical["source_id"]
            else:
                self.assertEqual(case["expected"], {"answer": None, "citations": []})
        self.assertGreater(lexical_disagreements, 10)
        self.assertGreater(len(winning_revisions), 20)

    def test_language_and_prompt_content_bind_case_identity(self):
        english = generate_cases(groups=1, languages=("en",))
        german = generate_cases(groups=1, languages=("de",))
        for key in ("id", "pair_id"):
            self.assertTrue({c[key] for c in english}.isdisjoint(c[key] for c in german))
        with patch.dict(_LANGUAGES["en"], {"system": _LANGUAGES["en"]["system"] + " Extra instruction."}):
            revised = generate_cases(groups=1, languages=("en",))
        self.assertTrue({c["id"] for c in english}.isdisjoint(c["id"] for c in revised))

    def test_no_question_revision_shortcuts_are_not_perfect(self):
        cases = generate_cases(groups=10)
        old_shortcut_correct = highest_revision_correct = 0
        for case in cases:
            supplied = records(case)
            old_marker = next((r for r in supplied if r["revision"] == 80), None)
            candidates = [r for r in supplied if r["revision"] in (10, 11)
                          and (old_marker is None or r["entity"] == old_marker["entity"])]
            old_winner = max(candidates, key=lambda r: r["revision"], default=None)
            old_answer = {"answer": old_winner["value"], "citations": [old_winner["source_id"]]} if old_winner else {
                "answer": None, "citations": []}
            old_shortcut_correct += old_answer == case["expected"]
            global_winner = max(supplied, key=lambda r: r["revision"])
            highest_revision_correct += case["expected"] == {
                "answer": global_winner["value"], "citations": [global_winner["source_id"]]}
        self.assertLess(old_shortcut_correct / len(cases), 0.6)
        self.assertLess(highest_revision_correct / len(cases), 0.6)

    def test_pair_interventions(self):
        for pair in pairs(generate_cases(groups=3)):
            before, after = pair["before"], pair["after"]
            self.assertEqual(question_target(before), question_target(after))
            first = {r["source_id"]: r for r in records(before)}
            second = {r["source_id"]: r for r in records(after)}
            if before["task"] == "update":
                self.assertNotEqual(before["expected"]["answer"], after["expected"]["answer"])
                self.assertEqual(before["expected"]["citations"], after["expected"]["citations"])
                changes = [source for source in first if first[source] != second[source]]
                self.assertEqual(changes, before["expected"]["citations"])
            elif before["task"] == "missing":
                self.assertIsNotNone(before["expected"]["answer"])
                self.assertEqual(after["expected"], {"answer": None, "citations": []})
                self.assertEqual(len(first), len(second))
                removed = set(first) - set(second)
                self.assertEqual(len(removed), 2)
                self.assertTrue(set(before["expected"]["citations"]).issubset(removed))
                self.assertNotIn(before["expected"]["answer"],
                                 [r["value"] for r in second.values()])
            else:
                self.assertEqual(before["expected"], after["expected"])
                self.assertEqual(set(first), set(second))
                changed = [source for source in first if first[source] != second[source]]
                self.assertEqual(len(changed), 1)
                bait = second[changed[0]]
                entity, field = question_target(after)
                self.assertEqual(bait["entity"], entity)
                self.assertNotEqual(bait["field"], field)
                self.assertGreater(bait["revision"], first[changed[0]]["revision"])
                self.assertNotEqual(bait["value"], after["expected"]["answer"])

    def test_language_subset_and_case_order(self):
        cases = generate_cases(groups=2, languages=("de",))
        self.assertEqual(len(cases), 12)
        self.assertEqual({c["language"] for c in cases}, {"de"})
        # Order is shuffled, rather than publishing a repeated before/after pattern.
        self.assertNotEqual([c["variant"] for c in cases], ["before", "after"] * 6)
        self.assertTrue(any(a["pair_id"] != b["pair_id"]
                            for a, b in zip(cases[::2], cases[1::2])))

    def test_invalid_settings(self):
        for options in ({"groups": 0}, {"groups": -1}, {"groups": True},
                        {"groups": 1.2}, {"seed": True}, {"seed": "42"},
                        {"languages": ()}, {"languages": ("en", "en")},
                        {"languages": ("fr",)}, {"languages": "en"},
                        {"languages": None}, {"languages": (["en"],)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                generate_cases(**options)


if __name__ == "__main__":
    unittest.main()
