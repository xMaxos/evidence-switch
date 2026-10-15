import json
import unittest
from candidate import prepare_retrieval
from evidenceswitch.generate import generate_cases
from evidenceswitch.selection import ANSWER_SCHEMA
from evidenceswitch.workflow import prepare


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.cases = generate_cases(seed=1107, groups=1)

    def test_exact_query_retrieves_all_revisions_without_sorting_or_value_substitution(self):
        for case in self.cases:
            prepared = prepare(case['messages'])
            target = prepared._records[0]
            query = {k: target[k] for k in ('entity', 'field')}
            final = prepare_retrieval(case['messages']).finish(json.dumps(query))
            records = json.loads(final[1]['content'][len(prepared._prefix):])
            self.assertEqual(records, [r for r in prepared._records
                if r['entity'] == target['entity'] and r['field'] == target['field']])
            self.assertEqual(len(records), 2)
            self.assertEqual(final[0]['content'], prepared._system+'\n\n'+ANSWER_SCHEMA[prepared._language])

    def test_model_query_is_not_corrected_using_question_or_gold(self):
        case = self.cases[0]
        prepared = prepare(case['messages'])
        candidate = prepare_retrieval(case['messages'])
        for query in ({'entity':'unknown', 'field':'unknown'},
                      {'entity':prepared._records[0]['entity']+' ', 'field':prepared._records[0]['field']}):
            final = candidate.finish(json.dumps(query))
            self.assertEqual(final[1]['content'], prepared._prefix+'[]')
        self.assertEqual(candidate.extraction_messages[1]['content'], prepared._prefix.split('\n\n')[0])
        for r in prepared._records:
            self.assertNotIn(r['value'], candidate.extraction_messages[1]['content'])
            self.assertNotIn(r['source_id'], candidate.extraction_messages[1]['content'])

    def test_malformed_query_is_rejected(self):
        candidate = prepare_retrieval(self.cases[0]['messages'])
        for raw in (None, '[]', 'null', '{}', '{"entity":"x"}',
                    '{"entity":"x","field":"y","answer":"cheat"}',
                    '{"entity":"x","entity":"y","field":"z"}',
                    '{"entity":"","field":"x"}', '{"entity":true,"field":"x"}',
                    '```json\n{"entity":"x","field":"y"}\n```'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                candidate.finish(raw)


if __name__ == '__main__':
    unittest.main()
