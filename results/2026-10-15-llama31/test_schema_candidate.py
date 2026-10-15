import copy
import json
import unittest

from candidate import prepare_ids
from schema_candidate import ANSWER_SCHEMA, prepare_schema_ids
from evidenceswitch.generate import generate_cases
from evidenceswitch.workflow import prepare


class SchemaCandidateTests(unittest.TestCase):
    def test_localized_schema_changes_only_final_system_message(self):
        for case in generate_cases(seed=707, groups=1):
            messages = case["messages"]
            baseline = prepare_ids(messages).finish('{"source_ids":[]}')
            result = prepare_schema_ids(messages).finish('{"source_ids":[]}')
            hint = ANSWER_SCHEMA[case["language"]]
            self.assertEqual(result, [
                {"role": "system", "content": baseline[0]["content"] + "\n\n" + hint},
                baseline[1]])
            self.assertIn('"citations":["', hint)
            self.assertIn('{"answer":null,"citations":[]}', hint)

    def test_extraction_is_identical_and_original_candidate_is_unchanged(self):
        for case in generate_cases(seed=707, groups=1):
            messages = case["messages"]
            original = copy.deepcopy(messages)
            ids = prepare_ids(messages)
            before = ids.finish('{"source_ids":[]}')
            schema = prepare_schema_ids(messages)
            self.assertEqual(schema.extraction_messages, ids.extraction_messages)
            schema.finish('{"source_ids":[]}')
            self.assertEqual(ids.finish('{"source_ids":[]}'), before)
            self.assertEqual(messages, original)

    def test_selected_records_remain_unchanged_and_invalid_ids_are_rejected(self):
        messages = generate_cases(seed=707, groups=1, languages=("en",))[0]["messages"]
        prepared = prepare(messages)
        records = json.loads(prepared._user[len(prepared._prefix):])
        selected = list(reversed(records))
        extraction = json.dumps({"source_ids": [record["source_id"] for record in selected]})
        schema = prepare_schema_ids(messages)
        final = schema.finish(extraction)
        self.assertEqual(final[1], prepare_ids(messages).finish(extraction)[1])
        self.assertEqual(json.loads(final[1]["content"][len(prepared._prefix):]), selected)
        with self.assertRaises(ValueError):
            schema.finish('{"source_ids":["invented"]}')


if __name__ == "__main__":
    unittest.main()
