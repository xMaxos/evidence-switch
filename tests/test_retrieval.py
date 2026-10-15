import json
import unittest
from evidenceswitch.generate import generate_cases
from evidenceswitch.retrieval import prepare_retrieval
from evidenceswitch.selection import ANSWER_SCHEMA
from evidenceswitch.workflow import prepare


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        system = generate_cases(groups=1, languages=('en',))[0]['messages'][0]
        self.records = [
            {'source_id':'wrong-field','entity':'Nuvora-0123456789ab','field':'beacon_key','revision':999,'value':'V-other-field'},
            {'source_id':'old','entity':'Nuvora-0123456789ab','field':'signal_code','revision':9,'value':'V-old'},
            {'source_id':'wrong-entity','entity':'Nuvora-0123456789ac','field':'signal_code','revision':9999,'value':'V-other-entity'},
            {'source_id':'new','entity':'Nuvora-0123456789ab','field':'signal_code','revision':10,'value':'V-new'},
        ]
        self.prefix = 'What is the field signal_code for entity Nuvora-0123456789ab?\n\nRecords:\n'
        self.messages = [system, {'role':'user','content':self.prefix+json.dumps(self.records)}]

    def records_after(self, query):
        final = prepare_retrieval(self.messages).finish(json.dumps(query))
        return json.loads(final[1]['content'][len(self.prefix):])

    def test_exact_retrieval_keeps_both_revisions_and_original_order(self):
        self.assertEqual(self.records_after({'entity':'Nuvora-0123456789ab','field':'signal_code'}),
                         [self.records[1], self.records[3]])

    def test_model_query_is_not_silently_corrected(self):
        self.assertEqual(self.records_after({'entity':'Nuvora-0123456789ac','field':'signal_code'}),
                         [self.records[2]])
        self.assertEqual(self.records_after({'entity':'Nuvora-0123456789ab ','field':'signal_code'}), [])
        self.assertEqual(self.records_after({'entity':'unknown','field':'signal_code'}), [])

    def test_query_has_question_only_and_answer_prompt_retains_schema(self):
        candidate = prepare_retrieval(self.messages)
        self.assertEqual(candidate.extraction_messages[1]['content'], self.prefix.split('\n\n')[0])
        for record in self.records:
            self.assertNotIn(record['value'], candidate.extraction_messages[1]['content'])
            self.assertNotIn(record['source_id'], candidate.extraction_messages[1]['content'])
        final = candidate.finish('{"entity":"Nuvora-0123456789ab","field":"signal_code"}')
        self.assertEqual(final[0]['content'], self.messages[0]['content']+'\n\n'+ANSWER_SCHEMA['en'])
        candidate.extraction_messages[1]['content'] = 'mutation'
        self.assertEqual(candidate.extraction_messages[1]['content'], self.prefix.split('\n\n')[0])
        for case in generate_cases(seed=1107, groups=1):
            prepared = prepare(case['messages'])
            final = prepare_retrieval(case['messages']).finish('{"entity":"unknown","field":"unknown"}')
            self.assertEqual(final[1]['content'], prepared._prefix+'[]')

    def test_malformed_query_fails(self):
        for raw in (None, '[]', 'null', '{}', '{"entity":"x"}',
                    '{"entity":"x","field":"y","answer":"cheat"}',
                    '{"entity":"x","entity":"y","field":"z"}',
                    '{"entity":"","field":"x"}', '{"entity":true,"field":"x"}',
                    '```json\n{"entity":"x","field":"y"}\n```'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                prepare_retrieval(self.messages).finish(raw)


if __name__ == '__main__':
    unittest.main()
