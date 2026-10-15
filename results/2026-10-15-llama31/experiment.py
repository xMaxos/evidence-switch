"""Local, sequential, paired comparison; never downloads or calls a cloud model."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time
from urllib.request import ProxyHandler, build_opener

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from evidenceswitch.generate import generate_cases
from evidenceswitch.io import read_jsonl, write_json, write_jsonl
from evidenceswitch.ollama import _request
from evidenceswitch.runner import _NoRedirects, _messages_digest
from evidenceswitch.workflow import EXTRACT, prepare
from candidate import EXTRACT_IDS, prepare_ids

MODEL = 'llama3.1:8b'
OPTIONS = {'temperature': 0, 'seed': 20261015, 'num_ctx': 4096, 'num_predict': 512}
STRATEGIES = ('direct', 'two-stage', 'two-stage-ids')
PHASES = {'dev': (707, 1), 'holdout': (808, 10)}
OPENER = build_opener(ProxyHandler({}), _NoRedirects())


def request(path, body=None):
    return _request(OPENER, 'http://127.0.0.1:11434' + path, body, 180)


def source_hashes():
    files = [HERE / 'candidate.py', HERE / 'experiment.py',
             ROOT / 'evidenceswitch/workflow.py', ROOT / 'evidenceswitch/generate.py']
    return {path.name: hashlib.sha256(path.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
            for path in files}


def environment():
    installed = next((m for m in request('/api/tags')['models'] if m['name'] == MODEL), None)
    if installed is None:
        raise ValueError('Required model is not already installed locally')
    details = request('/api/show', {'model': MODEL})
    if any(m.get('remote_host') or m.get('remote_model') for m in (installed, details)):
        raise ValueError('Cloud models are not permitted')
    return {'model': installed, 'ollama': request('/api/version')}


def freeze():
    frozen = dict(environment(), frozen_utc=datetime.now(timezone.utc).isoformat(),
        options=OPTIONS, think=False, api_fees_eur=0,
        implementation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        hardware={'gpu': 'NVIDIA GeForce RTX 4070 Ti', 'vram_mib': 12282},
        download='Llama 3.1 8B was downloaded once from the official Ollama registry; about 4.9 GB.',
        extract_instructions=EXTRACT, id_instructions=EXTRACT_IDS, sources_sha256=source_hashes(),
        phases={name: {'seed': seed, 'groups': groups, 'cases': groups * 18}
                for name, (seed, groups) in PHASES.items()},
        prompt_hashes={},
        primary_comparison='two-stage-ids versus two-stage; pair accuracy on 90 holdout pairs',
        transfer_comparison='two-stage versus direct; same 180 holdout cases',
        multiplicity='Two exact pair-level McNemar tests, Holm adjusted; other comparisons descriptive.',
        order='Cycle all six permutations across consecutive cases; each condition in each position equally often.',
        selection='Model-selected source ID order is preserved; code checks unique membership only, never relevance or revision.',
        rules='New conversation per call, identical per-call caps, no repair, no fallback, no selective model retries. Truncation counts as failure. Development excluded from holdout.',
        integration_gate='Add ID mode as an option only if holdout pair and strict accuracy do not fall versus two-stage, and either pair accuracy improves with fewer generated tokens, or generated tokens decrease at least 25% and median wall time at least 15%. Label unchanged quality as efficiency-only. Keep direct default; disclose individual regressions.',
        limitations='Same three synthetic task types, not new task families. One new quantized model. Candidate uses two calls versus one direct call.')
    for phase, (seed, groups) in PHASES.items():
        cases = generate_cases(seed=seed, groups=groups)
        prompts = [{k: c[k] for k in ('id', 'messages')} for c in cases]
        write_jsonl(HERE / f'{phase}-cases.jsonl', cases)
        write_jsonl(HERE / f'{phase}-prompts.jsonl', prompts)
        frozen['prompt_hashes'][phase] = _messages_digest(prompts)
    write_json(HERE / 'frozen.json', frozen)
    print('Frozen 18 development and 180 held-out cases, three strategies.', flush=True)


def complete(messages):
    started = time.perf_counter()
    response = request('/api/chat', {'model': MODEL, 'messages': messages, 'stream': False,
                       'think': False, 'keep_alive': '5m', 'options': OPTIONS})
    content = response.get('message', {}).get('content')
    if not isinstance(content, str) or response.get('done') is not True:
        raise RuntimeError('Invalid or incomplete local model response')
    stats = {key: response.get(key) for key in ('model', 'created_at', 'done', 'done_reason',
        'total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration')}
    stats.update(messages_sha256=_messages_digest(messages), wall_seconds=time.perf_counter() - started,
                 thinking_returned=bool(response['message'].get('thinking')), response=content)
    return content, stats


def evaluate(prompt, strategy):
    started = time.perf_counter()
    trace = {'id': prompt['id'], 'strategy': strategy, 'calls': [],
             'original_messages_sha256': _messages_digest(prompt['messages'])}
    def call(messages):
        content, stats = complete(messages)
        trace['calls'].append(stats)
        return content, stats['done_reason'] == 'stop'
    if strategy == 'direct':
        content, stopped = call(prompt['messages'])
        if not stopped:
            content = {'error': 'truncated_response'}
    else:
        prepared = (prepare if strategy == 'two-stage' else prepare_ids)(prompt['messages'])
        extraction, stopped = call(prepared.extraction_messages)
        trace.update(extraction_response=extraction, extraction_valid=False)
        try:
            if not stopped:
                raise ValueError('Truncated extraction')
            messages = prepared.finish(extraction)
        except (ValueError, RecursionError):
            content = {'error': 'invalid_extraction'}
        else:
            trace['extraction_valid'] = True
            content, stopped = call(messages)
            if not stopped:
                content = {'error': 'truncated_response'}
    trace['wall_seconds'] = time.perf_counter() - started
    return {'id': prompt['id'], 'response': content}, trace


def run(phase):
    frozen = json.loads((HERE / 'frozen.json').read_text(encoding='utf-8'))
    current = environment()
    if current['model']['digest'] != frozen['model']['digest'] or current['ollama'] != frozen['ollama']:
        raise ValueError('Model or Ollama changed after freezing')
    if frozen['sources_sha256'] != source_hashes() or frozen['options'] != OPTIONS:
        raise ValueError('Experiment code or options changed after freezing')
    prompts = read_jsonl(HERE / f'{phase}-prompts.jsonl')
    if _messages_digest(prompts) != frozen['prompt_hashes'][phase]:
        raise ValueError('Prompt file changed after freezing')
    for prompt in prompts:
        if set(prompt) != {'id', 'messages'}:
            raise ValueError('Answer keys or metadata in prompt input')
        prepare(prompt['messages'])
        prepare_ids(prompt['messages'])
    # Six cyclic permutations balance warm-cache and position effects.
    from itertools import permutations
    orders = list(permutations(STRATEGIES))
    started = time.perf_counter()
    with ExitStack() as stack:
        outputs = {name: stack.enter_context((HERE / f'{phase}-{name}.jsonl').open('x', encoding='utf-8', newline='\n'))
                   for name in STRATEGIES}
        traces = stack.enter_context((HERE / f'{phase}-traces.jsonl').open('x', encoding='utf-8', newline='\n'))
        for i, prompt in enumerate(prompts):
            for strategy in orders[i % len(orders)]:
                prediction, trace = evaluate(prompt, strategy)
                outputs[strategy].write(json.dumps(prediction, ensure_ascii=False) + '\n')
                outputs[strategy].flush()
                traces.write(json.dumps(trace, ensure_ascii=False) + '\n')
                traces.flush()
            if (i + 1) % 6 == 0 or i == 0:
                print(f'{phase}: {i+1}/{len(prompts)} cases, all three strategies, {time.perf_counter()-started:.1f}s', flush=True)
    print(f'{phase} complete: {len(prompts)} cases, all saved.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('freeze', *PHASES))
    phase = parser.parse_args().phase
    freeze() if phase == 'freeze' else run(phase)
