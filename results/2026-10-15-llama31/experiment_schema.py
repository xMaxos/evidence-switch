"""Development-only amendment: explicit answer shape, frozen before the main test."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import time

import experiment as base
from schema_candidate import ANSWER_SCHEMA, prepare_schema_ids

HERE = base.HERE
STRATEGIES = (*base.STRATEGIES, 'two-stage-ids-schema')


def source_hashes():
    files = ('experiment.py', 'candidate.py', 'experiment_schema.py', 'schema_candidate.py')
    return {name: hashlib.sha256((HERE / name).read_text(encoding='utf-8').encode()).hexdigest()
            for name in files}


def freeze():
    parent = json.loads((HERE / 'frozen.json').read_text(encoding='utf-8'))
    frozen = dict(parent, frozen_utc=datetime.now(timezone.utc).isoformat(),
        parent_frozen_sha256=hashlib.sha256((HERE / 'frozen.json').read_text(encoding='utf-8').encode()).hexdigest(),
        sources_sha256=source_hashes(), answer_hints=ANSWER_SCHEMA,
        amendment='After 18 development cases only: all six German cases failed output shape in every strategy. Add explicit localized citations-as-strings example to the final ID-mode system prompt. Retain unmodified ID mode as an ablation. Main seed808 has never been run.',
        phases={'dev-schema': {'seed': 707, 'groups': 1, 'cases': 18},
                'holdout': {'seed': 808, 'groups': 10, 'cases': 180}},
        prompt_hashes={'dev-schema': parent['prompt_hashes']['dev'], 'holdout': parent['prompt_hashes']['holdout']},
        order='Holdout cycles four rotations of direct, two-stage, two-stage-ids, two-stage-ids-schema. Each condition occupies each position 45 times.',
        primary_comparison='two-stage-ids-schema versus two-stage; pair accuracy on 90 holdout pairs',
        integration_gate='Add ID+schema mode as an option only if holdout pair and strict accuracy do not fall versus two-stage, and either pair accuracy improves with fewer generated tokens, or generated tokens decrease at least 25% and median wall time at least 15%. Label unchanged quality as efficiency-only. Keep direct default; disclose individual regressions.')
    for kind in ('cases', 'prompts'):
        base.write_jsonl(HERE / f'dev-schema-{kind}.jsonl', base.read_jsonl(HERE / f'dev-{kind}.jsonl'))
    base.write_json(HERE / 'schema-frozen.json', frozen)
    print('Amendment frozen before main test: four strategies, same 180 unseen cases.', flush=True)


def evaluate(prompt):
    started = time.perf_counter()
    trace = {'id': prompt['id'], 'strategy': 'two-stage-ids-schema', 'calls': [],
             'original_messages_sha256': base._messages_digest(prompt['messages']), 'extraction_valid': False}
    prepared = prepare_schema_ids(prompt['messages'])
    extraction, stats = base.complete(prepared.extraction_messages)
    trace['calls'].append(stats)
    trace['extraction_response'] = extraction
    try:
        if stats['done_reason'] != 'stop':
            raise ValueError('Truncated extraction')
        messages = prepared.finish(extraction)
    except (ValueError, RecursionError):
        content = {'error': 'invalid_extraction'}
    else:
        trace['extraction_valid'] = True
        content, stats = base.complete(messages)
        trace['calls'].append(stats)
        if stats['done_reason'] != 'stop':
            content = {'error': 'truncated_response'}
    trace['wall_seconds'] = time.perf_counter() - started
    return {'id': prompt['id'], 'response': content}, trace


def run(phase):
    frozen = json.loads((HERE / 'schema-frozen.json').read_text(encoding='utf-8'))
    env = base.environment()
    if env['model']['digest'] != frozen['model']['digest'] or env['ollama'] != frozen['ollama']:
        raise ValueError('Model or server changed after freezing')
    if source_hashes() != frozen['sources_sha256'] or base.OPTIONS != frozen['options']:
        raise ValueError('Frozen code or settings changed')
    prompts = base.read_jsonl(HERE / f'{phase}-prompts.jsonl')
    if base._messages_digest(prompts) != frozen['prompt_hashes'][phase]:
        raise ValueError('Frozen prompts changed')
    for prompt in prompts:
        if set(prompt) != {'id', 'messages'}:
            raise ValueError('Non-prompt fields present')
        prepare_schema_ids(prompt['messages'])
    strategies = ('two-stage-ids-schema',) if phase == 'dev-schema' else STRATEGIES
    start = time.perf_counter()
    with ExitStack() as stack:
        outputs = {name: stack.enter_context((HERE / f'{phase}-{name}.jsonl').open('x', encoding='utf-8', newline='\n'))
                   for name in strategies}
        traces = stack.enter_context((HERE / f'{phase}-traces.jsonl').open('x', encoding='utf-8', newline='\n'))
        for i, prompt in enumerate(prompts):
            offset = i % len(strategies)
            order = strategies[offset:] + strategies[:offset]
            for strategy in order:
                prediction, trace = evaluate(prompt) if strategy == 'two-stage-ids-schema' else base.evaluate(prompt, strategy)
                outputs[strategy].write(json.dumps(prediction, ensure_ascii=False) + '\n')
                outputs[strategy].flush()
                traces.write(json.dumps(trace, ensure_ascii=False) + '\n')
                traces.flush()
            if i == 0 or (i + 1) % 6 == 0:
                print(f'{phase}: {i+1}/{len(prompts)} cases, {len(strategies)} strategies, {time.perf_counter()-start:.1f}s', flush=True)
    print(f'{phase} complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('freeze', 'dev-schema', 'holdout'))
    phase = parser.parse_args().phase
    freeze() if phase == 'freeze' else run(phase)
