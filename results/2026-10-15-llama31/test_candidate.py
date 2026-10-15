import copy
import json
import unittest

from candidate import EXTRACT_IDS, prepare_ids
from evidenceswitch.generate import generate_cases
from evidenceswitch.workflow import prepare


class IdCandidateTests(unittest.TestCase):
    def setUp(self):
        self.messages = generate_cases(seed=707, groups=1, languages=("en",))[0]["messages"]
        self.original = copy.deepcopy(self.messages)
        self.prepared = prepare(self.messages)
        self.records = json.loads(self.prepared._user[len(self.prepared._prefix):])
        self.candidate = prepare_ids(self.messages)

    def test_all_languages_preserve_original_user_and_resolve_in_model_order(self):
        for case in generate_cases(seed=707, groups=1):
            messages = case["messages"]
            prepared = prepare(messages)
            records = json.loads(prepared._user[len(prepared._prefix):])
            selected = list(reversed(records[1:4]))
            candidate = prepare_ids(messages)
            self.assertEqual(candidate.extraction_messages, [
                {"role": "system", "content": EXTRACT_IDS[case["language"]]}, messages[1]])
            self.assertEqual(candidate.finish(json.dumps({
                "source_ids": [item["source_id"] for item in selected]})),
                prepared.finish(json.dumps({"records": selected})))

    def test_empty_selection_still_produces_answering_prompt(self):
        self.assertEqual(self.candidate.finish('{"source_ids":[]}'),
                         self.prepared.finish('{"records":[]}'))

    def test_all_records_including_irrelevant_ones_are_allowed_without_ranking(self):
        selected = list(reversed(self.records))
        self.assertEqual(self.candidate.finish(json.dumps({
            "source_ids": [item["source_id"] for item in selected]})),
            self.prepared.finish(json.dumps({"records": selected})))

    def test_invalid_id_extraction_is_rejected(self):
        first_id = self.records[0]["source_id"]
        invalid = [None, {}, "[]", "null", '```json\n{"source_ids":[]}\n```',
                   '{"source_ids":[],"source_ids":[]}', '{"source_ids":[],"extra":1}',
                   '{"source_ids":null}', '{"source_ids":{}}', '{"records":[]}',
                   '{"source_ids":[true]}', '{"source_ids":[1]}', '{"source_ids":[null]}',
                   '{"source_ids":[{}]}', '{"source_ids":[[]]}', '{"source_ids":[""]}',
                   '{"source_ids":["invented-source"]}',
                   json.dumps({"source_ids": [first_id, first_id]}),
                   json.dumps({"source_ids": [first_id, "invented-source"]})]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.candidate.finish(raw)

    def test_original_messages_are_unmodified_and_do_not_share_returned_objects(self):
        self.candidate.finish(json.dumps({"source_ids": [self.records[0]["source_id"]]}))
        self.assertEqual(self.messages, self.original)
        expected = self.candidate.extraction_messages
        emitted = self.candidate.extraction_messages
        emitted[0]["content"] = "changed"
        emitted[1]["content"] = "changed"
        self.messages[1]["content"] = "caller mutation"
        self.assertEqual(self.candidate.extraction_messages, expected)

    def test_bad_original_prompts_are_rejected_before_extraction(self):
        invalid = [None, [], list(reversed(self.messages))]
        changed = copy.deepcopy(self.messages)
        changed[0]["content"] += "extra instruction"
        invalid.append(changed)
        changed = copy.deepcopy(self.messages)
        changed[1]["content"] = self.prepared._prefix + json.dumps([self.records[0], self.records[0]])
        invalid.append(changed)
        changed = copy.deepcopy(self.messages)
        changed[1]["expected"] = {}
        invalid.append(changed)
        for messages in invalid:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                prepare_ids(messages)


if __name__ == "__main__":
    unittest.main()
