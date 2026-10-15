# Numeric revision selection: regression fix and fresh tests

[All experiments](../README.md) · [Project overview](../../README.md)

The two known Llama failures were reproduced and corrected by selecting the
unique highest **integer** revision in the retrieval tool. The comparison is
against v0.4.0 `two-stage-retrieval`, with identical model-query instructions and
final-answer format. The new optional strategy is `two-stage-latest`.

## Known-error regression check

These are reused diagnostic fixtures, excluded from fresh-test scores:

| Known failure | Old revision chosen | Correct revision | New Llama result |
| --- | ---: | ---: | --- |
| `a9de83eee7b1` (German) | 7616 | 7624 | Correct |
| `8c879a46804c` (Spanish) | 8961 | 9313 | Correct |

Both full pairs were rerun: Llama improved from 2/4 to 4/4; Qwen remained 4/4.
The 18-case development set, seed 1307, scored 18/18 for both strategies on
both models. No candidate change was made after seeing these results.

## Fresh holdout

| Model | Prior retrieval | Latest-revision mode | Correct answer pairs |
| --- | ---: | ---: | ---: |
| llama | 179/180 | 180/180 | 89/90 → 90/90 |
| qwen | 180/180 | 180/180 | 90/90 → 90/90 |

| Model | Correct answer and source | Strict final JSON |
| --- | ---: | ---: |
| llama | 178/180 → 180/180 | 178/180 → 180/180 |
| qwen | 180/180 → 180/180 | 180/180 → 180/180 |

- llama: 1 fixed answers, 0 regressions; exact answer-pair p=1, Holm-adjusted p=1.
- qwen: 0 fixed answers, 0 regressions; exact answer-pair p=1, Holm-adjusted p=1.

On the fresh set, Llama's answer-value error `1d1571b5afc8` and additional
source error `74064aa7f865` were corrected. Neither model regressed on any case.
The primary adjusted p-values are 1.0: this small difference does **not**
establish a statistically significant improvement. Qwen was already at ceiling.

Detailed per-case values, grounded-content scores, strict format scores and
costs are in [Llama analysis](holdout-llama-analysis.json) and
[Qwen analysis](holdout-qwen-analysis.json). The aggregate adoption gate passed;
that gate separately checks the two known errors and no aggregate loss on the
new data. Known-error success is not counted as fresh-test improvement.

The [plan](holdout-plan.json) was frozen before inference: seed 1408, 180 new
cases, 90 pairs, with update, missing-evidence and distractor tasks in English,
German and Spanish. Both models received the **same** 180 questions. These are
360 model/task evaluations, not 360 different questions. Each condition came
first in 90 cases and second in 90. There were no retries, answer repairs,
fallbacks or holdout-based tuning. Fixed inference seeds do not guarantee
bitwise-identical output.

## What changed

The model extracts the entity and field from the question. Code performs exact
matching and selects the unique highest numeric revision. The same model then
copies the value and cites the selected source. The record is unchanged; no
answer key or programmatic final answer is sent back as a prediction.

If no record matches, the model still receives an empty list and generates its
answer. A tie at the highest matching revision fails with `invalid_extraction`
instead of choosing a source arbitrarily. Lower-revision ties are harmless.
Ambiguity and numeric-boundary behavior have software tests; the synthetic
model benchmark itself has unique highest revisions.

**The tool now implements both matching and revision ranking.** On this
benchmark the model's remaining tasks are query extraction and value/citation
copying or abstention. This is an engineering fix to the combined workflow,
not improved model weights or proof of independent reasoning. It should not
be advertised as a model beating the benchmark unaided. Fresh seeds use the
same templates and do not establish generalization to new task families,
natural documents, semantic retrieval, or longer contexts.

## Scoring and cost

The unchanged primary metric checks the exact answer value from parseable JSON,
independent of citation shape and extra keys. Missing/unparseable answers fail.
Grounded content additionally requires the expected source identifiers;
strict JSON scores are reported separately. Pair-level exact McNemar tests
are Holm-adjusted across the two models. Small or ceiling-level differences
do not establish a population-level accuracy gain.

| Model / method | Calls | Median seconds/case | Input tokens | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| llama: v0.4 retrieval | 360 | 0.698 | 89,029 | 9,009 |
| llama: v0.5 latest | 360 | 0.689 | 79,220 | 8,968 |
| qwen: v0.4 retrieval | 360 | 1.024 | 93,962 | 9,808 |
| qwen: v0.5 latest | 360 | 1.010 | 81,285 | 9,744 |

Hosted API fees: **EUR 0**. Both installed models ran locally with Ollama 0.35.0
on an RTX 4070 Ti; no model downloads were needed. Electricity and hardware use
still apply. Models were `llama3.1:8b` and `qwen3.5:9b`, both Q4_K_M; exact
digests are in the plan. Settings: `temperature=0`, `seed=20261015`, `num_ctx=4096`,
`num_predict=512` per call, `think=false`, no server-enforced JSON schema.
Both methods use two calls per completed case but different input lengths;
compute is not exactly matched. Timings are machine-specific.

## Inspect and reproduce

All raw predictions, request traces, frozen plans, inputs, source snapshots and
analysis scripts are kept in this directory. `regression-*` contains reused
known-error pairs, `dev-*` the small development set, and `holdout-*` the fresh
test. Verify without model calls:

```sh
python results/2026-10-15-latest-revision/verify.py
```

For an independent new run, follow the [usage guide](../../docs/usage.md),
generate a fresh seed, and compare `two-stage-retrieval` with `two-stage-latest`
on the same prompts. Use new output paths. Sequential CLI runs reproduce the
workflow, not the historical balanced timing schedule. Do not tune on these
published holdout cases and then label them unseen.
