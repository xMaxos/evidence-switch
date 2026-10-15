# Experiment index

[Project overview](../README.md) · [Methodology](../docs/methodology.md) · [Changelog](../CHANGELOG.md)

These are local synthetic tests, not evidence of greater general intelligence.
Each row uses its own baseline and fixtures. **Do not compare percentages across
different seeds as a version leaderboard.** Both model families use a single
quantized version. All model runs used zero hosted API fees.

## Published comparisons

| Version | Report | Main comparison and scope |
| --- | --- | --- |
| 0.5 | [Numeric revision selection](2026-10-15-latest-revision/) | v0.4 retrieval vs highest-revision retrieval; 180 new questions per model, plus separate known-error and development checks |
| 0.4 | [Exact retrieval](2026-10-15-exact-retrieval/) | ID selection vs model-generated exact queries; 180 new questions per model; Llama 158→178 and Qwen 178→180 |
| 0.3 | [IDs and format guidance](2026-10-15-llama31/) | Llama four-condition test on 180 questions; full-copy 112→refined IDs 167, much of the gain from format; separate 36-case Qwen check |
| 0.2 | [Full-record selection](2026-10-15-qwen35-two-stage/) | Qwen direct vs two-stage; 54 primary +18 conditional confirmation cases; descriptive total 66→72 |

## Rejected development attempts

| Attempt | Outcome | Scope |
| --- | --- | --- |
| [Complete boolean audit](2026-10-15-content-audit/) | Rejected: Llama 15→1, Qwen 18→12 correct | Same 18 development questions; incomplete or malformed audits |
| [Relevance-only view](2026-10-15-relevance-view/) | Stopped: Llama 15→15 | Same 18 development questions; not pursued on Qwen or a holdout |

The v0.2 report also preserves its earlier exploratory attempts. Development
sets can be reused to design changes; they are never counted as independent
confirmation. Known-error regression cases likewise remain separate from fresh
holdout tests. Historical filenames and raw responses are kept unchanged for
traceability, including the v0.3 plain-ID ablation's historical strategy name.

## Verify everything offline

From the repository root:

```sh
python scripts/verify_results.py
```

No model requests or network access are made. Each published comparison also
has its own `verify.py`. Reports link their frozen plans, raw outputs, request
hashes and cost measurements. Software tests and artifact verification are
checks of implementation and recorded evidence, not additional AI experiments.
