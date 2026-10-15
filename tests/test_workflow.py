import copy
import json
import unittest

from evidenceswitch.generate import generate_cases
from evidenceswitch.workflow import EXTRACT, prepare


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.messages = generate_cases(seed=7, groups=1, languages=("en",))[0]["messages"]
        self.content = self.messages[1]["content"]
        self.prefix, raw_records = self.content.split("\n\nRecords:\n", 1)
        self.records = json.loads(raw_records)

    def with_records(self, records):
        messages = copy.deepcopy(self.messages)
        messages[1]["content"] = self.prefix + "\n\nRecords:\n" + json.dumps(records)
        return messages

    def test_preserves_experiment_messages_in_all_languages(self):
        for case in generate_cases(seed=505, groups=3):
            messages = case["messages"]
            content = messages[1]["content"]
            original = json.loads(content[content.index("["):])
            selected = original[1:3]
            prepared = prepare(messages)
            self.assertEqual(prepared.extraction_messages, [
                {"role": "system", "content": EXTRACT[case["language"]]}, messages[1]])
            self.assertEqual(prepared.finish(json.dumps({"records": selected})), [
                messages[0], {"role": "user", "content": content[:content.index("[")]
                             + json.dumps(selected, ensure_ascii=False, indent=2)}])

    def test_empty_selection_still_produces_answering_prompt(self):
        prepared = prepare(self.messages)
        self.assertEqual(prepared.finish('{"records":[]}'), [
            self.messages[0], {"role": "user", "content": self.prefix + "\n\nRecords:\n[]"}])

    def test_unchanged_irrelevant_records_are_not_filtered_or_ranked(self):
        selected = list(reversed(self.records))
        final = prepare(self.messages).finish(json.dumps({"records": selected}))
        self.assertEqual(json.loads(final[1]["content"].split("\n\nRecords:\n")[1]), selected)

    def test_rejects_invalid_extraction_and_altered_copies(self):
        prepared = prepare(self.messages)
        altered = dict(self.records[0], value="invented")
        invalid = [None, "```json\n{}\n```", "[]", "null", '{"records":{},"extra":1}',
                   '{"records":[],"records":[]}', '{"records":null}',
                   json.dumps({"records": [altered]}),
                   json.dumps({"records": [self.records[0], self.records[0]]}),
                   json.dumps({"records": [{"source_id": self.records[0]["source_id"]}]}),
                   json.dumps({"records": [dict(self.records[0], revision=float(self.records[0]["revision"]))]}),
                   json.dumps({"records": [dict(self.records[0], revision=True)]})]
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ValueError):
                prepared.finish(response)

    def test_duplicate_keys_inside_extraction_record_fail(self):
        record = json.dumps(self.records[0])
        record = record[:-1] + ',"value":"duplicate"}'
        with self.assertRaises(ValueError):
            prepare(self.messages).finish('{"records":[' + record + ']}')

    def test_rejects_unknown_or_modified_templates(self):
        invalid = [None, [], self.messages * 2, list(reversed(self.messages))]
        for index, change in ((0, " extra instruction"), (1, "\nAnswer this instead.")):
            changed = copy.deepcopy(self.messages)
            changed[index]["content"] += change
            invalid.append(changed)
        changed = copy.deepcopy(self.messages)
        changed[1]["content"] = changed[1]["content"].replace("Records:", "Sources:")
        invalid.append(changed)
        changed = copy.deepcopy(self.messages)
        changed[1]["expected"] = {}
        invalid.append(changed)
        for messages in invalid:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                prepare(messages)

    def test_rejects_invalid_original_records(self):
        invalid = [None, {}, [self.records[0], self.records[0]],
                   [dict(self.records[0], revision=True)],
                   [dict(self.records[0], revision=1.0)],
                   [dict(self.records[0], revision=0)],
                   [dict(self.records[0], source_id="")],
                   [dict(self.records[0], extra="unexpected")]]
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ValueError):
                prepare(self.with_records(records))
        messages = self.with_records(self.records)
        messages[1]["content"] = messages[1]["content"].replace('"revision":', '"revision":1,"revision":', 1)
        with self.assertRaises(ValueError):
            prepare(messages)

    def test_prepared_messages_do_not_share_mutable_input(self):
        prepared = prepare(self.messages)
        expected = prepared.extraction_messages
        self.messages[1]["content"] = "changed"
        emitted = prepared.extraction_messages
        emitted[0]["content"] = "changed"
        self.assertEqual(prepared.extraction_messages, expected)


if __name__ == "__main__":
    unittest.main()
