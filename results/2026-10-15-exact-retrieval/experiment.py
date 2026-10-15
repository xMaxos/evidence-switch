"""Frozen local comparison of ID selection and complete relevance auditing."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from evidenceswitch.generate import generate_cases
from evidenceswitch.io import read_jsonl, write_json, write_jsonl
from evidenceswitch.ollama import _request
from evidenceswitch.runner import _NoRedirects, _messages_digest
from evidenceswitch.selection import prepare_ids
from candidate import prepare_retrieval

MODELS = {'llama': 'llama3.1:8b', 'qwen': 'qwen3.5:9b'}
STRATEGIES = {'two-stage-ids': prepare_ids, 'two-stage-retrieval': prepare_retrieval}
OPTIONS = {'temperature': 0, 'seed': 20261015, 'num_ctx': 4096, 'num_predict': 512}
PHASES = {'dev': (1107, 1), 'holdout': (1208, 10)}
OPENER = build_opener(ProxyHandler({}), _NoRedirects())


def request(path, body=None):
    return _request(OPENER, 'http://127.0.0.1:11434' + path, body, 180)


def source_hashes():
    paths = [HERE / name for name in ('candidate.py', 'experiment.py', 'analyze.py')]
    paths += [ROOT / 'evidenceswitch' / name for name in ('generate.py', 'workflow.py', 'selection.py', 'score.py')]
    return {p.name: hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest() for p in paths}


def environment():
    tags = request('/api/tags')['models']
    result = {}
    for alias, model in MODELS.items():
        found = next((m for m in tags if m['name'] == model), None)
        if found is None:
            raise ValueError('Required model is not installed; no automatic downloads')
        details = request('/api/show', {'model': model})
        if any(item.get('remote_host') or item.get('remote_model') for item in (found, details)):
            raise ValueError('Cloud models forbidden')
        result[alias] = {'name': model, 'digest': found['digest'], 'details': found['details']}
    return {'models': result, 'ollama': request('/api/version')['version']}


def freeze(phase):
    seed, groups = PHASES[phase]
    cases = generate_cases(seed=seed, groups=groups)
    prompts = [{key: c[key] for key in ('id', 'messages')} for c in cases]
    if phase == 'holdout':
        for alias in MODELS:
            if not (HERE / f'dev-{alias}-analysis.json').exists():
                raise ValueError('Review both development results before freezing holdout')
    plan = dict(environment(), frozen_utc=datetime.now(timezone.utc).isoformat(),
        phase=phase, seed=seed, groups=groups, cases=len(cases), languages=['en', 'de', 'es'],
        options=OPTIONS, think=False, api_fees_eur=0,
        implementation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        sources_sha256=source_hashes(), prompt_hash=_messages_digest(prompts),
        strategies=list(STRATEGIES), model_order=list(MODELS) if phase == 'dev' else list(reversed(MODELS)),
        order='Alternate baseline/candidate and candidate/baseline on successive cases; fresh conversations. Same cases for both models.',
        primary_metric='Exact answer value, ignoring citation shape and extra JSON keys. Require parseable JSON object with answer of correct type; no answer repair. Compare both-correct pairs; exact two-sided McNemar, Holm over two models.',
        secondary='Grounded content: exact answer and expected citation IDs, accepting string IDs or singleton source_id objects; strict JSON score and missing-evidence errors reported separately.',
        gate='Adopt as optional only if answer-value, grounded-content, and strict correct counts and answer-pair counts do not fall on either model, with at least one model improving answer-value count. Report each individual regression. No holdout tuning.',
        workflow='Model extracts entity and field from question only. Code executes exact equality retrieval using that model query, keeping all matching records and original order. Model chooses revision and final answer. Same final schema as baseline. No answer keys and no code-based ranking or final-answer substitution.',
        rules='No gold or pair metadata in requests, no retries, no output repair. Truncation is failure. Development excluded from holdout. Fixed settings do not guarantee bitwise determinism.',
        limitations='Same synthetic templates, one quantization per model, one machine; two calls per condition. The exact retriever implements the structured task matching rule; gains belong to the hybrid workflow, not intrinsic model intelligence. No general intelligence or training claim.')
    write_jsonl(HERE / f'{phase}-cases.jsonl', cases)
    write_jsonl(HERE / f'{phase}-prompts.jsonl', prompts)
    write_json(HERE / f'{phase}-plan.json', plan)
    print(f'Frozen {phase}: {len(cases)} cases per model.', flush=True)


def evaluate(prompt, strategy, model):
    started = time.perf_counter()
    trace = {'id': prompt['id'], 'strategy': strategy, 'calls': [],
             'original_messages_sha256': _messages_digest(prompt['messages']), 'extraction_valid': False}

    def call(messages):
        begin = time.perf_counter()
        result = request('/api/chat', {'model': model, 'messages': messages, 'stream': False,
                         'think': False, 'keep_alive': '5m', 'options': OPTIONS})
        content = result.get('message', {}).get('content')
        if not isinstance(content, str) or result.get('done') is not True:
            raise RuntimeError('Incomplete local response; completed evidence preserved')
        stats = {k: result.get(k) for k in ('model', 'created_at', 'done', 'done_reason', 'total_duration',
                 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration')}
        stats.update(response=content, messages_sha256=_messages_digest(messages),
                     wall_seconds=time.perf_counter()-begin, thinking_returned=bool(result['message'].get('thinking')))
        trace['calls'].append(stats)
        return content, stats['done_reason'] == 'stop'

    prepared = STRATEGIES[strategy](prompt['messages'])
    extraction, stopped = call(prepared.extraction_messages)
    trace['extraction_response'] = extraction
    try:
        if not stopped:
            raise ValueError('Truncated audit')
        messages = prepared.finish(extraction)
    except (ValueError, RecursionError):
        content = {'error': 'invalid_extraction'}
    else:
        trace['extraction_valid'] = True
        content, stopped = call(messages)
        if not stopped:
            content = {'error': 'truncated_response'}
    trace['wall_seconds'] = time.perf_counter()-started
    return {'id': prompt['id'], 'response': content}, trace


def run(phase, alias):
    plan = json.loads((HERE / f'{phase}-plan.json').read_text(encoding='utf-8'))
    current = environment()
    if current['models'] != plan['models'] or current['ollama'] != plan['ollama']:
        raise ValueError('Model/server differs from frozen plan')
    if source_hashes() != plan['sources_sha256'] or OPTIONS != plan['options']:
        raise ValueError('Frozen code or settings changed')
    prompts = read_jsonl(HERE / f'{phase}-prompts.jsonl')
    if _messages_digest(prompts) != plan['prompt_hash']:
        raise ValueError('Frozen inputs changed')
    for prompt in prompts:
        if set(prompt) != {'id', 'messages'}:
            raise ValueError('Non-prompt data in model inputs')
        for prepare in STRATEGIES.values():
            prepare(prompt['messages'])
    begin = time.perf_counter()
    prefix = f'{phase}-{alias}'
    with ExitStack() as stack:
        outputs = {s: stack.enter_context((HERE / f'{prefix}-{s}.jsonl').open('x', encoding='utf-8', newline='\n')) for s in STRATEGIES}
        traces = stack.enter_context((HERE / f'{prefix}-traces.jsonl').open('x', encoding='utf-8', newline='\n'))
        for i, prompt in enumerate(prompts):
            order = list(STRATEGIES)[::1 if i % 2 == 0 else -1]
            for strategy in order:
                prediction, trace = evaluate(prompt, strategy, MODELS[alias])
                outputs[strategy].write(json.dumps(prediction, ensure_ascii=False)+'\n')
                outputs[strategy].flush()
                traces.write(json.dumps(trace, ensure_ascii=False)+'\n')
                traces.flush()
            if i == 0 or (i+1) % 12 == 0 or i+1 == len(prompts):
                print(f'{prefix}: {i+1}/{len(prompts)}, {time.perf_counter()-begin:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'run'))
    parser.add_argument('phase', choices=PHASES)
    parser.add_argument('model', choices=MODELS, nargs='?')
    args = parser.parse_args()
    freeze(args.phase) if args.action == 'freeze' else run(args.phase, args.model)
