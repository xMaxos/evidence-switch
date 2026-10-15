"""Recompute saved content results and check the integrated workflow offline."""
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import analyze
from evidenceswitch.io import parse_json, read_jsonl
from evidenceswitch.runner import _messages_digest as digest


def same(actual, saved, path='report'):
    if type(actual) is not type(saved):
        raise ValueError('Changed type: '+path)
    if isinstance(actual, dict):
        if set(actual) != set(saved):
            raise ValueError('Changed keys: '+path)
        for key in actual:
            same(actual[key], saved[key], path+'.'+key)
    elif isinstance(actual, list):
        if len(actual) != len(saved):
            raise ValueError('Changed length: '+path)
        for i, (a, b) in enumerate(zip(actual, saved)):
            same(a, b, path+'.'+str(i))
    elif isinstance(actual, float):
        if not math.isclose(actual, saved, rel_tol=1e-12, abs_tol=0):
            raise ValueError('Changed number: '+path)
    elif actual != saved:
        raise ValueError('Changed value: '+path)


def summary():
    reports = {alias: analyze.analyze('holdout', alias) for alias in ('llama','qwen')}
    comparisons = {alias: report['comparison']['answer_correct'] for alias,report in reports.items()}
    p_values = sorted((v['pair_exact_mcnemar_p'], k) for k,v in comparisons.items())
    adjusted, previous = {}, 0
    for rank, (p, alias) in enumerate(p_values):
        previous = max(previous, min(1.0, (len(p_values)-rank)*p))
        adjusted[alias] = previous
    gate_keys = ('answer_correct','grounded_correct','strict_correct','answer_pairs_correct')
    checks, improved = {}, False
    for alias, report in reports.items():
        old = report['strategies']['two-stage-retrieval']
        new = report['strategies']['two-stage-latest']
        checks[alias] = {key: new[key] >= old[key] for key in gate_keys}
        improved |= new['answer_correct'] > old['answer_correct']
    from caseset import FAILURE_IDS
    regression = analyze.analyze('regression', 'llama')['strategies']['two-stage-latest']['results']
    fixed_known = all(next(row for row in regression if row['id'] == case_id)['strict_correct'] for case_id in FAILURE_IDS)
    return {'cases_per_model': 180, 'model_count': 2, 'api_fees_eur': 0,
            'answer_pair_holm_adjusted_p': adjusted,
            'integration_gate': {'per_model_nondecrease': checks, 'at_least_one_fresh_answer_gain': improved, 'both_known_llama_errors_fixed': fixed_known,
                                 'passes': fixed_known and all(all(v.values()) for v in checks.values())}}


def production_replay():
    from evidenceswitch.latest import prepare_latest
    cases = {c['id']: c for c in read_jsonl(HERE / 'holdout-cases.jsonl')}
    count = 0
    for alias in ('llama','qwen'):
        traces = [t for t in read_jsonl(HERE / f'holdout-{alias}-traces.jsonl') if t['strategy']=='two-stage-latest']
        analyze.require(len(traces) == 180, 'Incomplete production replay')
        for trace in traces:
            prepared = prepare_latest(cases[trace['id']]['messages'])
            messages = [prepared.extraction_messages]
            try:
                if trace['calls'][0]['done_reason'] != 'stop':
                    raise ValueError('Truncated')
                final = prepared.finish(trace['extraction_response'])
            except (ValueError, RecursionError):
                analyze.require(trace['extraction_valid'] is False, 'Production rejection differs')
            else:
                analyze.require(trace['extraction_valid'] is True, 'Production acceptance differs')
                messages.append(final)
            analyze.require(len(messages) == len(trace['calls']), 'Production call count differs')
            for message, call in zip(messages, trace['calls']):
                analyze.require(digest(message) == call['messages_sha256'], 'Production request differs')
                count += 1
    return count


def check_content_metric():
    case = {'id':'test','pair_id':'pair','language':'en','task':'update',
            'expected':{'answer':'V-right','citations':['source']}}
    shapes = ['{"answer":"V-right","citations":["source"]}',
              '{"answer":"V-right","citations":[{"source_id":"source"}],"extra":1}']
    for raw in shapes:
        result = analyze.content_result(case, raw)
        analyze.require(result['answer_correct'] and result['grounded_correct'], 'Content depends on citation shape')
    wrong = analyze.content_result(case, '{"answer":"V-wrong","citations":["source"]}')
    analyze.require(not wrong['answer_correct'] and not wrong['grounded_correct'], 'Wrong value accepted')
    unsupported = analyze.content_result(case, '{"answer":"V-right","citations":["wrong"]}')
    analyze.require(unsupported['answer_correct'] and not unsupported['grounded_correct'], 'Unsupported answer accepted')
    absent = dict(case, expected={'answer':None, 'citations':[]})
    analyze.require(not analyze.content_result(absent, '{}')['answer_correct'], 'Missing answer mistaken for null')


def main():
    check_content_metric()
    for phase in ('regression','dev','holdout'):
        for alias in ('llama','qwen'):
            saved = parse_json((HERE/f'{phase}-{alias}-analysis.json').read_text(encoding='utf-8'))
            same(analyze.analyze(phase, alias), saved)
            print(f'Verified {phase}/{alias}: original values, raw outputs, message hashes and costs.')
    same(summary(), parse_json((HERE/'summary.json').read_text(encoding='utf-8')))
    print(f'Production matches {production_replay()} measured request hashes. No model calls.')


if __name__ == '__main__':
    main()
