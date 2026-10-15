# Source-ID selection and an explicit answer format

Local experiment, 15 October 2026. On 180 new Llama 3.1 cases, the refined
workflow improved strict accuracy and reduced runtime versus full-record
selection. Much of the accuracy gain was correct output formatting. This does
not establish a general reasoning improvement.

## Results and naming

Correct means the exact answer and citation in the required JSON shape. A pair
passes only if both independently presented variants are correct.

| Method in raw experiment files | Correct / 180 | Pairs / 90 | Production CLI name |
| --- | ---: | ---: | --- |
| `direct` | 82 (45.6%) | 25 | `direct` (default) |
| `two-stage`: full-record selection | 112 (62.2%) | 43 | `two-stage` |
| `two-stage-ids`: plain ID selection ablation | 117 (65.0%) | 46 | Not exposed |
| `two-stage-ids-schema`: IDs plus final format reminder | **167 (92.8%)** | **77** | **`two-stage-ids`** |

**Production `two-stage-ids` means the refined ID-plus-format workflow**, not the
plain-ID ablation bearing that name in the historical raw files.

Against the existing full-record workflow, the refined method fixed **56 cases
and regressed one**, case `dd5b4c174317`. Thirty-five pairs improved and one
regressed. The primary pair-level exact McNemar p-value was 1.08e-9; Holm adjustment
over the two planned comparisons gave 2.15e-9. This describes this fixed
synthetic sample, not broad model competence. Full-record selection also beat
direct on aggregate, but introduced ten individual regressions.

## What changed, and what the gain means

The first call asks the model for all matching source IDs, including older and
newer revisions. Code checks unique membership and restores the original records
in the model-selected order. It never decides relevance or the winning revision.
The second call answers independently from those records and includes an
explicit example of `citations` as strings rather than objects. The example
contains placeholders, not answers. No answer keys enter either request.

The refined workflow had valid extraction in all 180 cases, but 34 extractions
included irrelevant records. None omitted a matching record. Thus valid ID
selection is not proof of relevance. Thirteen final answers still failed:
11 of the 30 cases with missing evidence, plus two revision-selection errors.
In the single regression, the old method correctly returned null; the refined
method selected a foreign entity's record and returned a wrong answer.
Full-record selection had seven invalid extractions; those failed without a
second call, repair, fallback, or selective retry.

A supplementary diagnostic only converted citation items of exactly
`{"source_id":"..."}` to strings. It did not change answers, invent citations,
strip code fences, or alter the actual strict scores. Under that limited
normalization, full-record correctness rises from **112 to 151/180**, while
the refined method remains **167/180**. Most of the strict-score gap therefore
reflects format compliance; a smaller difference remains in this diagnostic.
It is not a clean causal decomposition or proof of better general reasoning.
The [diagnostic plan](content-diagnostic-plan.json) was specified while the main
run was in progress, before inspecting its outputs.

Plain IDs and refined IDs used identical extraction prompts, but eight of 180
raw extraction responses differed, including three different selected sets,
despite fixed inference seeds. Their comparison does not isolate the format
reminder with perfectly identical intermediate inputs; exact determinism is
not assumed.

## Cost and settings

| Method | Model calls | Median seconds/case | Output tokens | Input tokens |
| --- | ---: | ---: | ---: | ---: |
| Direct | 180 | 0.535 | 5,812 | 121,236 |
| Full records | 353 | 2.347 | 31,306 | 173,429 |
| Plain IDs | 360 | 0.806 | 9,591 | 172,636 |
| IDs plus format reminder | 360 | 0.699 | 8,534 | 192,011 |

Versus full records, the refined method used **72.7% fewer output tokens** and
had **70.2% lower median wall time**. Input tokens increased by 10.7%. These are
not claims of equal percentage reductions in total tokens or energy. Direct
was faster still. Timings are specific to this machine and request schedule.

- Model: `llama3.1:8b`, 8.0B, `Q4_K_M`, Ollama `0.35.0`.
- Digest: `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`.
- Hardware: NVIDIA GeForce RTX 4070 Ti, 12,282 MiB reported VRAM.
- Every call: `temperature=0`, `seed=20261015`, `num_ctx=4096`,
  `num_predict=512`, `think=false`; no server-enforced JSON format.
- All 1,253 holdout requests ended with `stop`, with no output truncation or
  thinking output. Development calls are excluded from these totals.

The two-stage methods allow two requests and up to 1,024 output tokens per case,
versus one request and 512 tokens for direct. This is not a compute-matched
comparison. Hosted API fees were **EUR 0**. Llama was downloaded once from the
official Ollama registry (about 4.9 GB); inference ran locally. Electricity,
storage, and download bandwidth still apply. Model weights are not bundled here.

## Development, frozen test, and limitations

Development used 18 cases, seed 707. Direct, full records, and plain IDs scored
5, 10, and 9 correct. All six German development cases failed output shape in
each method. The explicit final format reminder was added using only that
development evidence; rerunning the same development cases yielded 15/18.
That rerun is exploratory, not independent confirmation.

The amended method and adoption rule were frozen before running the 180-case
holdout, seed 808, ten groups per task and language. Four condition orders were
rotated so each method occupied each position 45 times. Calls started fresh
conversations; there were no selective model retries. The two planned
comparisons were refined IDs versus full records, and full records versus
direct. Other comparisons are exploratory ablations.

The frozen integration gate required no aggregate strict or pair decrease
versus full records, plus a stated quality/efficiency benefit. It passed, but
aggregate improvement did not mean zero individual regressions. The option was
added without changing the default. Generator tasks remain update, missing,
and distractor in English, German, and Spanish. These are fresh seeds from the
same templates, not held-out task families. This test covers one fixed
quantized Llama model and one machine; it does not establish performance on
real documents, other model versions, open-ended tasks, or learning.

## Qwen compatibility check

After the Llama adoption gate passed, the prespecified check ran both production
workflows on 36 fresh `qwen3.5:9b` cases, seed 909. **Both scored 36/36 cases and
18/18 pairs, with no regressions.** This is compatibility evidence at a ceiling,
not a demonstrated accuracy gain, and is separate from the Llama primary result.

Each method used 72 calls. Median wall time fell from 2.594 to 1.235 seconds
(52.4%); output tokens fell from 5,151 to 2,030 (60.6%). Input tokens rose from
40,712 to 44,800, so total input-plus-output tokens increased slightly, from
45,863 to 46,830. The old and new conditions ran sequentially, making these
timings descriptive. All calls stopped normally without thinking output.
The model digest and settings are in the [frozen compatibility plan](qwen-regression-plan.json);
[qwen-analysis.json](qwen-analysis.json) contains the full comparison.

## Inspect, verify, and rerun

From the repository root, this audits saved evidence without model calls,
including the diagnostic and production-workflow compatibility comparison:

```sh
python results/2026-10-15-llama31/verify.py
```

Evidence includes the [original plan](frozen.json), [schema amendment](schema-frozen.json),
[holdout inputs](holdout-cases.jsonl), [raw direct responses](holdout-direct.jsonl),
[raw full-record responses](holdout-two-stage.jsonl),
[plain-ID ablation](holdout-two-stage-ids.jsonl),
[refined responses](holdout-two-stage-ids-schema.jsonl),
[request traces](holdout-traces.jsonl), [analysis](holdout-analysis.json),
and [content diagnostic](holdout-content-diagnostic.json).
The earlier [Qwen experiment](../2026-10-15-qwen35-two-stage/) remains unchanged.

To run a new comparison with an already installed local model:

```sh
python -c "from pathlib import Path; Path('runs').mkdir(exist_ok=True)"
python -m evidenceswitch generate --out data/recheck-808 --seed 808 --groups 10
python -m evidenceswitch run-ollama --prompts data/recheck-808/prompts.jsonl --model llama3.1:8b --strategy two-stage --out runs/recheck-old.jsonl --trace runs/recheck-old-trace.jsonl
python -m evidenceswitch run-ollama --prompts data/recheck-808/prompts.jsonl --model llama3.1:8b --strategy two-stage-ids --out runs/recheck-ids.jsonl --trace runs/recheck-ids-trace.jsonl
python -m evidenceswitch compare --cases data/recheck-808/cases.jsonl --baseline runs/recheck-old.jsonl --candidate runs/recheck-ids.jsonl --out runs/recheck-comparison.json
```

Choose new output paths for each run. These commands use the measured inference
defaults but sequential conditions, unlike the historical balanced schedule;
timings are not directly comparable. Fixed seeds do not guarantee identical
inference across software and hardware. Use fresh seeds and additional models
to investigate generalization rather than tuning on these published cases.
