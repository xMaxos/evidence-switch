"""Known failing pairs are regression fixtures, never fresh efficacy evidence."""
from pathlib import Path
from evidenceswitch.generate import generate_cases
from evidenceswitch.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
FAILURE_IDS = ('8c879a46804c', 'a9de83eee7b1')
PHASES = {'regression': (None, 0), 'dev': (1307, 1), 'holdout': (1408, 10)}


def build_cases(phase):
    if phase != 'regression':
        seed, groups = PHASES[phase]
        return generate_cases(seed=seed, groups=groups)
    original = read_jsonl(ROOT / 'results/2026-10-15-exact-retrieval/holdout-cases.jsonl')
    pairs = {case['pair_id'] for case in original if case['id'] in FAILURE_IDS}
    if len(pairs) != 2:
        raise ValueError('Known failing pairs are missing')
    return [case for case in original if case['pair_id'] in pairs]
