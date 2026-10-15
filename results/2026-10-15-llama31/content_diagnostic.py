"""Supplementary format diagnostic; never replaces strict benchmark scores."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from evidenceswitch.io import parse_json, read_jsonl, write_json
from evidenceswitch.score import score, valid_answer

STRATEGIES = {
    'dev': ('direct', 'two-stage', 'two-stage-ids'),
    'dev-schema': ('two-stage-ids-schema',),
    'holdout': ('direct', 'two-stage', 'two-stage-ids', 'two-stage-ids-schema'),
}
LIMITATION = ('Supplementary counterfactual content score, not accepted actual success. '
              'Only citation objects with exactly one string source_id are normalized. '
              'This does not establish improved reasoning or change the primary metric or integration gate.')


def normalize_citation_objects(raw):
    if not isinstance(raw, str):
        return raw, False
    try:
        value = parse_json(raw)
    except (ValueError, RecursionError):
        return raw, False
    if (not isinstance(value, dict) or set(value) != {'answer', 'citations'}
            or not isinstance(value['citations'], list)):
        return raw, False
    changed = False
    citations = []
    for item in value['citations']:
        if isinstance(item, dict) and set(item) == {'source_id'} and isinstance(item['source_id'], str):
            item, changed = item['source_id'], True
        citations.append(item)
    normalized = {'answer': value['answer'], 'citations': citations}
    if not changed or not valid_answer(normalized):
        return raw, False
    return json.dumps(normalized, ensure_ascii=False), True


def analyze_content(phase):
    cases = read_jsonl(HERE / f'{phase}-cases.jsonl')
    reports = {}
    for strategy in STRATEGIES[phase]:
        rows = read_jsonl(HERE / f'{phase}-{strategy}.jsonl')
        strict = score(cases, rows)
        if strict['summary']['responses'] != len(cases):
            raise ValueError('Diagnostic requires complete predictions; main test may still be running')
        normalized, normalized_ids = [], []
        for row in rows:
            response, changed = normalize_citation_objects(row['response'])
            normalized.append({'id': row['id'], 'response': response})
            if changed:
                normalized_ids.append(row['id'])
        content = score(cases, normalized)
        extra = [after['id'] for before, after in zip(strict['results'], content['results'])
                 if after['correct'] and not before['correct']]
        reports[strategy] = {
            'cases': len(cases), 'strict_correct': sum(r['correct'] for r in strict['results']),
            'content_correct': sum(r['correct'] for r in content['results']),
            'responses_normalized': len(normalized_ids), 'normalized_ids': normalized_ids,
            'extra_correct_after_normalization': len(extra), 'extra_correct_ids': extra,
            'by_language': {lang: {'strict_correct': sum(r['correct'] for r in strict['results'] if r['language'] == lang),
                                   'content_correct': sum(r['correct'] for r in content['results'] if r['language'] == lang)}
                            for lang in sorted({c['language'] for c in cases})}}
    return {'phase': phase, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'interpretation': LIMITATION, 'strategies': reports}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=LIMITATION)
    parser.add_argument('phase', choices=tuple(STRATEGIES))
    phase = parser.parse_args().phase
    write_json(HERE / f'{phase}-content-diagnostic.json', analyze_content(phase))
