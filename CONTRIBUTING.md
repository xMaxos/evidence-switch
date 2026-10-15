# Contributing

[Project overview](README.md) · [Methodology](docs/methodology.md)

```sh
python -m unittest discover -s tests -v
```

Tests use the standard library; runner tests start a temporary local HTTP server.
No model or API key is needed. GitHub Actions is configured for Python 3.10,
3.12, and 3.13 on Linux and Windows.

Useful contributions include small reproducible failures, new language templates,
and fixtures that remove answer shortcuts. Include the seed and expected behavior.
Keep fixture generation and scoring deterministic, add a regression test, and
separate model prompts from answer keys. Do not submit private data or API keys.

Verify every published result, including unsuccessful development attempts:

```sh
python scripts/verify_results.py
```

Keep published raw evidence immutable. Add a new result bundle with frozen
settings, raw outputs and a verifier; distinguish development, known-error
regression and fresh holdout data. Document any deterministic rule moved from
the model into code. Do not claim improved general intelligence from these tasks.
