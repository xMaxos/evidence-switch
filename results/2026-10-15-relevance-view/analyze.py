"""Offline scoring: content values are assessed independently of output shape."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from evidenceswitch.generate import generate_cases
from evidenceswitch.io import parse_json, read_jsonl, write_json
from evidenceswitch.runner import _messages_digest as digest
from evidenceswitch.score import score
from evidenceswitch.selection import prepare_ids
from candidate import prepare_focus

# Independent question/record gold computation, already published and audited.
spec = importlib.util.spec_from_file_location('prior_analysis', ROOT / 'results/2026-10-15-llama31/analyze.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
require = prior.require
PREPARE = {'two-stage-ids': prepare_ids, 'two-stage-focus': prepare_focus}


def content_result(case, raw):
    try:
        value = parse_json(raw) if isinstance(raw, str) else raw
    except (ValueError, RecursionError):
        value = None
    answer_valid = isinstance(value, dict) and 'answer' in value and (value['answer'] is None or isinstance(value['answer'], str))
    answer = answer_valid and value['answer'] == case['expected']['answer']
    cited = None
    if isinstance(value, dict) and isinstance(value.get('citations'), list):
        ids = [item['source_id'] if isinstance(item, dict) and set(item) == {'source_id'} else item for item in value['citations']]
        if all(isinstance(item, str) and item for item in ids):
            cited = set(ids)
    return {'id': case['id'], 'pair_id': case['pair_id'], 'language': case['language'], 'task': case['task'],
            'answer_correct': bool(answer), 'grounded_correct': bool(answer and cited == set(case['expected']['citations']))}


def pair_flags(rows, key):
    pairs = defaultdict(list)
    for row in rows:
        pairs[row['pair_id']].append(row[key])
    return {pair: all(flags) for pair, flags in pairs.items()}


def comparison(before, after, key):
    old, new = ({r['id']: r[key] for r in rows} for rows in (before, after))
    fixed = [k for k in old if not old[k] and new[k]]
    regressed = [k for k in old if old[k] and not new[k]]
    oldp, newp = pair_flags(before, key), pair_flags(after, key)
    fp = [k for k in oldp if not oldp[k] and newp[k]]
    rp = [k for k in oldp if oldp[k] and not newp[k]]
    n = len(fp) + len(rp)
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(len(fp), len(rp))+1))/2**n)
    return {'fixed_ids': fixed, 'regressed_ids': regressed, 'pair_fixed_ids': fp,
            'pair_regressed_ids': rp, 'pair_exact_mcnemar_p': p}


def analyze(phase, alias):
    plan = parse_json((HERE / f'{phase}-plan.json').read_text(encoding='utf-8'))
    cases = read_jsonl(HERE / f'{phase}-cases.jsonl')
    require(cases == generate_cases(seed=plan['seed'], groups=plan['groups']), 'Regenerated cases differ')
    prompts = [{k: c[k] for k in ('id', 'messages')} for c in cases]
    require(read_jsonl(HERE / f'{phase}-prompts.jsonl') == prompts and digest(prompts) == plan['prompt_hash'], 'Prompt mismatch')
    require(len(cases) == plan['cases'], 'Wrong case count')
    for c in cases:
        require(prior.gold_from_sources(c)[0] == c['expected'], 'Independent gold mismatch')
    if phase == 'holdout':
        dev = read_jsonl(HERE / 'dev-cases.jsonl')
        require(not {c['id'] for c in dev} & {c['id'] for c in cases}, 'Development overlap')
    source_paths = {p.name: p for p in [HERE / n for n in ('candidate.py', 'experiment.py', 'analyze.py')] + [ROOT / 'evidenceswitch' / n for n in ('generate.py', 'workflow.py', 'selection.py', 'score.py')]}
    for name, expected in plan['sources_sha256'].items():
        require(hashlib.sha256(source_paths[name].read_text(encoding='utf-8').encode()).hexdigest() == expected, 'Frozen source changed: '+name)
    prefix = f'{phase}-{alias}'
    traces = read_jsonl(HERE / f'{prefix}-traces.jsonl')
    order = [(c['id'], s) for i, c in enumerate(cases) for s in list(PREPARE)[::1 if i % 2 == 0 else -1]]
    require([(t['id'], t['strategy']) for t in traces] == order, 'Incomplete or reordered calls')
    result, outcomes = {}, {}
    for strategy, prepare in PREPARE.items():
        predictions = read_jsonl(HERE / f'{prefix}-{strategy}.jsonl')
        require([r['id'] for r in predictions] == [c['id'] for c in cases], 'Incomplete predictions')
        strict = score(cases, predictions)
        outcomes[strategy] = rows = [dict(content_result(c, p['response']), strict_correct=s['correct']) for c, p, s in zip(cases, predictions, strict['results'])]
        calls, walls, issues, invalid = [], [], [], []
        for case, pred, trace in zip(cases, predictions, [t for t in traces if t['strategy'] == strategy]):
            require(trace['original_messages_sha256'] == digest(case['messages']), 'Original prompt changed')
            prepared = prepare(case['messages'])
            messages = [prepared.extraction_messages]
            recorded = trace['calls']
            require(recorded[0]['response'] == trace['extraction_response'], 'Extraction differs')
            try:
                if recorded[0]['done_reason'] != 'stop':
                    raise ValueError('Truncated')
                final = prepared.finish(trace['extraction_response'])
            except (ValueError, RecursionError):
                require(trace['extraction_valid'] is False and pred['response'] == {'error': 'invalid_extraction'}, 'Invalid extraction not failed')
                invalid.append(case['id'])
            else:
                require(trace['extraction_valid'] is True, 'Valid extraction marked invalid')
                messages.append(final)
                require(pred['response'] == (recorded[-1]['response'] if recorded[-1]['done_reason'] == 'stop' else {'error': 'truncated_response'}), 'Raw answer altered')
                selected = {r['source_id'] for r in prior.records_from(final)[1]}
                relevant = prior.gold_from_sources(case)[1]
                if selected != relevant:
                    issues.append({'id': case['id'], 'missing': sorted(relevant-selected), 'extra': sorted(selected-relevant)})
            require(len(messages) == len(recorded), 'Retry or missing call')
            for message, call in zip(messages, recorded):
                require(digest(message) == call['messages_sha256'], 'Request hash changed')
                require(call['model'] == plan['models'][alias]['name'] and call['done'] is True and call['thinking_returned'] is False, 'Model or thinking differs')
                for key in ('eval_count', 'prompt_eval_count', 'load_duration'):
                    require(type(call[key]) is int and call[key] >= 0, 'Invalid cost')
                require(math.isfinite(call['wall_seconds']) and call['wall_seconds'] >= 0, 'Invalid duration')
            require(math.isfinite(trace['wall_seconds']) and trace['wall_seconds'] >= 0, 'Invalid case duration')
            calls.extend(recorded)
            walls.append(trace['wall_seconds'])
        result[strategy] = {'answer_correct': sum(r['answer_correct'] for r in rows),
            'grounded_correct': sum(r['grounded_correct'] for r in rows), 'strict_correct': sum(r['strict_correct'] for r in rows),
            'answer_pairs_correct': sum(pair_flags(rows, 'answer_correct').values()),
            'strict_pairs_correct': sum(pair_flags(rows, 'strict_correct').values()),
            'format_valid': sum(r['format_valid'] for r in strict['results']),
            'missing_evidence_correct': sum(r['answer_correct'] for r, c in zip(rows, cases) if c['expected']['answer'] is None),
            'missing_evidence_cases': sum(c['expected']['answer'] is None for c in cases),
            'by_language': {lang: {key: sum(r[key] for r in rows if r['language'] == lang) for key in ('answer_correct','grounded_correct','strict_correct')} for lang in ('en','de','es')},
            'cost': {'requests': len(calls), 'output_tokens': sum(c['eval_count'] for c in calls), 'input_tokens': sum(c['prompt_eval_count'] for c in calls),
                     'median_seconds': statistics.median(walls), 'total_seconds': sum(walls),
                     'non_stop_requests': sum(c['done_reason'] != 'stop' for c in calls), 'max_output_tokens': max(c['eval_count'] for c in calls)},
            'extraction': {'invalid_ids': invalid, 'irrelevance_or_omission': issues}, 'results': rows}
    old, new = (outcomes[s] for s in PREPARE)
    return {'phase': phase, 'model': alias, 'cases': len(cases), 'pairs': len(cases)//2, 'strategies': result,
            'comparison': {key: comparison(old, new, key) for key in ('answer_correct', 'grounded_correct', 'strict_correct')},
            'audit': {'independent_gold': len(cases), 'raw_calls_and_hashes': len([c for t in traces for c in t['calls']]), 'balanced_order': True}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('dev','holdout'))
    parser.add_argument('model', choices=('llama','qwen'))
    args = parser.parse_args()
    report = analyze(args.phase, args.model)
    write_json(HERE / f'{args.phase}-{args.model}-analysis.json', report)
    print(json.dumps({k: {n:v[n] for n in ('answer_correct','grounded_correct','strict_correct','answer_pairs_correct','cost')} for k,v in report['strategies'].items()}))
