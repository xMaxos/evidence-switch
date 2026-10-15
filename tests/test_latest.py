import copy
import json
import unittest
from evidenceswitch.generate import generate_cases
from evidenceswitch.latest import prepare_latest
from evidenceswitch.retrieval import prepare_retrieval


class LatestTests(unittest.TestCase):
    def setUp(self):
        self.system = generate_cases(groups=1, languages=('en',))[0]['messages'][0]
        self.prefix = 'What is the field signal_code for entity Nuvora-0123456789ab?\n\nRecords:\n'
        self.query = '{"entity":"Nuvora-0123456789ab","field":"signal_code"}'

    def record(self, source, revision, **changes):
        return dict({'source_id':source, 'entity':'Nuvora-0123456789ab', 'field':'signal_code',
                     'revision':revision, 'value':'V-'+source}, **changes)

    def messages(self, records):
        return [self.system, {'role':'user', 'content':self.prefix+json.dumps(records)}]

    def test_numeric_revision_boundaries_and_both_input_orders(self):
        for older, newer in ((9,10),(99,100),(7616,7624),(8961,9313),(999,1000)):
            old, new = self.record('old',older), self.record('new',newer)
            unrelated = self.record('irrelevant',99999, field='beacon_key')
            for records in ([old,unrelated,new],[new,unrelated,old]):
                with self.subTest(older=older, records=records):
                    messages = self.messages(records)
                    saved = copy.deepcopy(messages)
                    final = prepare_latest(messages).finish(self.query)
                    self.assertEqual(json.loads(final[1]['content'][len(self.prefix):]), [new])
                    self.assertEqual(messages, saved)
                    self.assertEqual(final[0], prepare_retrieval(messages).finish(self.query)[0])

    def test_only_highest_ties_fail_without_arbitrary_source_choice(self):
        low = self.record('low',9)
        tie = self.record('tie',9)
        high = self.record('high',10)
        prepared = prepare_latest(self.messages([low,tie,high]))
        self.assertEqual(json.loads(prepared.finish(self.query)[1]['content'][len(self.prefix):]), [high])
        for records in ([low,tie], [low,high,self.record('high2',10)]):
            with self.subTest(records=records), self.assertRaisesRegex(ValueError, 'unique'):
                prepare_latest(self.messages(records)).finish(self.query)

    def test_no_match_and_wrong_model_query_are_not_corrected(self):
        other = self.record('other',12, entity='Nuvora-0123456789ac')
        prepared = prepare_latest(self.messages([self.record('right',10),other]))
        self.assertEqual(prepared.finish('{"entity":"unknown","field":"signal_code"}')[1]['content'], self.prefix+'[]')
        final = prepared.finish('{"entity":"Nuvora-0123456789ac","field":"signal_code"}')
        self.assertEqual(json.loads(final[1]['content'][len(self.prefix):]), [other])

    def test_query_call_and_final_schema_unchanged_in_all_languages(self):
        for case in generate_cases(seed=1307,groups=1):
            latest = prepare_latest(case['messages'])
            baseline = prepare_retrieval(case['messages'])
            self.assertEqual(latest.extraction_messages, baseline.extraction_messages)
            query = '{"entity":"unknown","field":"unknown"}'
            self.assertEqual(latest.finish(query), baseline.finish(query))
        with self.assertRaises(ValueError):
            prepare_latest(self.messages([self.record('x',1)])).finish('{"entity":"missing-field"}')


if __name__ == '__main__':
    unittest.main()
