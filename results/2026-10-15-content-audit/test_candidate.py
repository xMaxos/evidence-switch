import copy
import json
import unittest
from candidate import prepare_audit
from evidenceswitch.generate import generate_cases
from evidenceswitch.workflow import prepare


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.messages = generate_cases(seed=1107, groups=1)[0]['messages']
        self.prepared = prepare(self.messages)
        self.ids = [r['source_id'] for r in self.prepared._records]

    def response(self, flags):
        return json.dumps({'checks': [[sid, *flags[i]] for i, sid in enumerate(self.ids)]})

    def selected(self, flags):
        final = prepare_audit(self.messages).finish(self.response(flags))
        return json.loads(final[1]['content'][len(self.prepared._prefix):])

    def test_two_independent_model_judgments_are_required(self):
        flags = [(True, False), (False, True), (False, False), (True, True)] * 2
        self.assertEqual(self.selected(flags), [self.prepared._records[3], self.prepared._records[7]])

    def test_all_true_preserves_even_irrelevant_and_older_records(self):
        # This detects an accidental code-based relevance filter or answer oracle.
        self.assertEqual(self.selected([(True, True)] * len(self.ids)), self.prepared._records)
        self.assertEqual(self.selected([(False, False)] * len(self.ids)), [])

    def test_incomplete_invented_duplicate_reordered_or_nonboolean_audits_fail(self):
        valid = [[sid, True, False] for sid in self.ids]
        cases = [valid[:-1], valid + [valid[0]], list(reversed(valid))]
        for change in (['invented', True, False], [self.ids[1], True, False],
                       [self.ids[0], 1, False], [self.ids[0], 'true', False],
                       [self.ids[0], True, None], [self.ids[0], True],
                       [self.ids[0], True, False, 'extra']):
            cases.append([change] + valid[1:])
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                prepare_audit(self.messages).finish(json.dumps({'checks': rows}))
        for raw in ('[]', '{"checks":null}', '{"checks":[],"extra":1}',
                    '{"checks":[],"checks":[]}', None):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                prepare_audit(self.messages).finish(raw)

    def test_same_final_format_no_mutation_and_all_languages(self):
        from evidenceswitch.selection import prepare_ids
        for case in generate_cases(seed=1107, groups=1):
            original = copy.deepcopy(case['messages'])
            prepared = prepare(original)
            ids = [r['source_id'] for r in prepared._records]
            rows = [[sid, i % 2 == 0, True] for i, sid in enumerate(ids)]
            selected = ids[::2]
            audit = prepare_audit(original)
            self.assertEqual(audit.finish(json.dumps({'checks': rows})),
                             prepare_ids(original).finish(json.dumps({'source_ids': selected})))
            audit.extraction_messages[1]['content'] = 'mutation'
            self.assertEqual(audit.extraction_messages[1], original[1])
            self.assertEqual(original, case['messages'])


if __name__ == '__main__':
    unittest.main()
