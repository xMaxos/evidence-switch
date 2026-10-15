"""Verify every published experiment locally; no model, network or API key."""
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RELEASES = ('2026-10-15-qwen35-two-stage', '2026-10-15-llama31',
            '2026-10-15-exact-retrieval', '2026-10-15-latest-revision')
DEVELOPMENT = {'2026-10-15-content-audit': ('llama', 'qwen'),
               '2026-10-15-relevance-view': ('llama',)}


def same(actual, saved):
    if type(actual) is not type(saved):
        raise ValueError('Saved result type changed')
    if isinstance(actual, dict):
        if set(actual) != set(saved):
            raise ValueError('Saved result keys changed')
        for key in actual:
            same(actual[key], saved[key])
    elif isinstance(actual, list):
        if len(actual) != len(saved):
            raise ValueError('Saved result length changed')
        for left, right in zip(actual, saved):
            same(left, right)
    elif isinstance(actual, float):
        if not math.isclose(actual, saved, rel_tol=1e-12, abs_tol=0):
            raise ValueError('Saved result number changed')
    elif actual != saved:
        raise ValueError('Saved result changed')


def main():
    # Each historical bundle has its own frozen candidate module. Separate
    # interpreters avoid importing a different bundle's same-named module.
    for folder in RELEASES:
        subprocess.run([sys.executable, str(ROOT/'results'/folder/'verify.py')], check=True, cwd=ROOT)
    code = """
import json, pathlib, sys
folder = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(folder))
import analyze
model = sys.argv[2]
actual = analyze.analyze('dev', model)
saved = json.loads((folder / ('dev-' + model + '-analysis.json')).read_text(encoding='utf-8'))
print(json.dumps({'actual':actual, 'saved':saved}, ensure_ascii=True))
"""
    for folder, models in DEVELOPMENT.items():
        for model in models:
            output = subprocess.check_output([sys.executable, '-c', code, str(ROOT/'results'/folder), model], cwd=ROOT, text=True)
            reports = json.loads(output)
            same(reports['actual'], reports['saved'])
            print(f'Verified rejected development attempt: {folder}/{model}.', flush=True)
    print('All published experiments verified offline. No model calls.', flush=True)


if __name__ == '__main__':
    main()
