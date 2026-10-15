# Model-generated queries with exact evidence retrieval

Local experiment, 15 October 2026. This changes content reliability through a
hybrid model/tool workflow. It does not train a model or prove greater general
intelligence. The baseline is version 0.3.0's **two-stage-ids**, including its
existing format guidance; it is not the older full-record copying baseline.

## Fresh holdout results

| Model | Existing: correct values | Retrieval: correct values | Both-correct answer pairs | Valid final format |
| --- | ---: | ---: | ---: | ---: |
| llama | 158/180 | 178/180 | 69/90 → 88/90 | 178/180 → 180/180 |
| qwen | 178/180 | 180/180 | 88/90 → 90/90 | 179/180 → 180/180 |

- llama: 20 newly correct cases, 0 regressions; missing-evidence answers 14/30 → 30/30. Answer-pair exact McNemar p=3.8147e-06; Holm-adjusted p=7.62939e-06.
- qwen: 2 newly correct cases, 0 regressions; missing-evidence answers 28/30 → 30/30. Answer-pair exact McNemar p=0.5; Holm-adjusted p=0.5.

Grounded-content and strict correct counts equal the answer-value counts in
this table. The raw predictions and individual fixes/regressions are saved in
[Llama analysis](holdout-llama-analysis.json) and [Qwen analysis](holdout-qwen-analysis.json).
Qwen's small two-case improvement does not establish a statistically reliable
population-level gain. One fix was an actual unsupported answer on missing
evidence (`333903b9961e`); the other was a failed source-ID extraction
(`72d7dde61863`). It would be misleading to call both reasoning errors.

Of Llama's 20 fixes, **18 replaced wrong answer values in already valid final
JSON**; two recovered failed source-ID extractions. The final format guidance
was unchanged, so those 18 are content fixes, not citation-shape repairs.
Two older-revision errors remained (`8c879a46804c`, `a9de83eee7b1`). Neither
model had an individual answer-value or strict-score regression in this sample.

## What the tool does

1. The model extracts the requested `entity` and `field` from the question alone.
2. Code retrieves records where both fields exactly equal that **model-generated
   query**, keeping all revisions and the original source order.
3. The model selects the revision, answer value and citation from those records.

The final answer instructions and schema are identical to the baseline. The
retriever implements the structured task's matching rule; it does not compute
the highest revision or substitute a final answer. Wrong model queries are not
silently corrected using the question or answer key. Unknown queries retrieve
an empty list, which is still passed to the model for answering. Malformed
queries fail without repair or retry. No gold answers reach either request.

This is an improvement to the **combined system**, with an exact retrieval
component doing work that the baseline left to the model. It is not evidence
that an unaided model became intrinsically smarter. These exact-match record
tasks are particularly suitable for a deterministic search tool.

## Content versus formatting

The primary score checks the exact answer value from a parseable JSON object,
ignoring citation shape and extra keys. Missing/unparseable answers still fail.
The separate grounded-content score also requires the correct citation IDs,
accepting strings or singleton `source_id` objects. Strict JSON validity is
reported separately. No inference output is repaired. Thus the primary metric
is not a pure measure of reasoning, but cannot be improved merely by changing
a citation object into a string while leaving the same wrong answer.

## Test design and unsuccessful attempts

Development reused 18 cases, seed 1107. A [complete boolean audit](../2026-10-15-content-audit/)
failed badly on both models. A [relevance-only view](../2026-10-15-relevance-view/)
did not improve Llama and was stopped before Qwen or holdout testing. Both
attempts and their raw outputs remain public. Exact retrieval then scored
18/18 on both development models, versus baseline 15/18 for Llama and 18/18
for Qwen. These repeated development results are tuning evidence only.

The final method, metrics and adoption gate were [frozen](holdout-plan.json)
before the main run: seed 1208, 180 fresh questions in English, German and
Spanish, 90 before/after pairs. Both models received the **same** 180 questions;
this is 360 model/task evaluations, not 360 different questions. Each strategy
occupied first and second position 90 times per model. No holdout tuning,
selective retries or fallback was allowed. Two primary answer-pair tests were
Holm-adjusted across models. The aggregate adoption gate passed; individual
regressions remain separately reported. The CLI default is still `direct`.

## Runtime and costs

| Model / method | Calls | Median seconds/case | Input tokens | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| llama: Existing IDs | 358 | 0.769 | 192,548 | 8,587 |
| llama: Exact retrieval | 360 | 0.687 | 89,343 | 9,009 |
| qwen: Existing IDs | 359 | 1.232 | 224,291 | 10,322 |
| qwen: Exact retrieval | 360 | 1.006 | 94,146 | 9,877 |

All inference ran locally through Ollama 0.35.0, with **EUR 0 hosted API fees**
and no model downloads during this experiment. Electricity and hardware use
still apply. Settings were `temperature=0`, `seed=20261015`, `num_ctx=4096`,
`num_predict=512` per call, `think=false`, with no server-enforced JSON schema.
Models: Llama 3.1 8B and Qwen 3.5 9B, both Q4_K_M; exact digests are in the plan.
The machine used an RTX 4070 Ti. Timings are machine-specific; seeds do not
guarantee bitwise-identical inference. Both methods use up to two calls, but
their input/output lengths differ, so compute is not exactly matched.

The task family is unchanged: synthetic updates, missing evidence and
distractors, with exact identifiers and values. These results do not establish
performance on natural documents, semantic search, long contexts, new task
families, other model versions or autonomous learning.

## Verify or rerun

From the repository root, verify scores, costs, input hashes, source hashes and
production request equivalence without running a model:

```sh
python results/2026-10-15-exact-retrieval/verify.py
```

To run a fresh local comparison with an already installed model:

```sh
python -m evidenceswitch generate --out data/content-recheck --seed 1208 --groups 10
python -c "from pathlib import Path; Path('runs').mkdir(exist_ok=True)"
python -m evidenceswitch run-ollama --prompts data/content-recheck/prompts.jsonl --model llama3.1:8b --strategy two-stage-ids --out runs/content-baseline.jsonl --trace runs/content-baseline-trace.jsonl
python -m evidenceswitch run-ollama --prompts data/content-recheck/prompts.jsonl --model llama3.1:8b --strategy two-stage-retrieval --out runs/content-retrieval.jsonl --trace runs/content-retrieval-trace.jsonl
python -m evidenceswitch compare --cases data/content-recheck/cases.jsonl --baseline runs/content-baseline.jsonl --candidate runs/content-retrieval.jsonl
```

Repeat with `--model qwen3.5:9b` and new output paths. These sequential CLI
commands reproduce the workflow, not the historical balanced timing schedule.
Use fresh seeds for new validation rather than tuning on this published set.
The ordinary CLI comparison uses strict scoring; `analyze.py` contains the
separately frozen content metrics for this experiment.
