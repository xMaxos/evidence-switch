# Two-stage evidence selection: local model results

Measured on 15 October 2026 with an existing local `qwen3.5:9b` model. The two-stage
workflow improved final answer accuracy on these samples. This is an observation
about a narrow workflow, not evidence of generally greater intelligence.

## Results

Correct means the exact answer **and** source identifier. A pair passes only when
both independently presented variants pass.

| Fixed test set | Direct correct | Two-stage correct | Direct pairs | Two-stage pairs |
| --- | ---: | ---: | ---: | ---: |
| Primary, seed 505 | 50/54 (92.6%) | 54/54 (100%) | 23/27 | 27/27 |
| Conditional confirmation, seed 606 | 16/18 (88.9%) | 18/18 (100%) | 7/9 | 9/9 |
| Descriptive total | 66/72 (91.7%) | 72/72 (100%) | 30/36 | 36/36 |

Six cases improved; none regressed. Both workflows correctly abstained in all 12
cases with no matching evidence. The exact two-sided McNemar p-values on pairs
were **0.125** for the primary test and **0.5** for confirmation. Neither small
test establishes a general improvement. The total is descriptive, not an
additional independent significance test.

The confirmation rule was frozen in advance: run the unchanged method if primary
pair accuracy improves. Thus the confirmation sample was conditionally run.
All examples use the same synthetic task family, three languages, and one
quantized model. New seeds do not establish performance on new task families.
Other models, real documents, open-ended questions, and learning were not tested.

## Method and its limits

1. The model selects all records matching the question's entity and field,
   retaining both older and newer matching revisions.
2. Code checks only that selected records are unchanged members of the supplied
   input, with no duplicate source identifiers. It does not determine relevance
   or choose the highest revision.
3. The same model receives the original question and instructions with the
   selected records, in a fresh conversation, and produces the final answer.

No answer keys, pair metadata, or scores enter either model request. Invalid
extractions count as failures, without repair, fallback, or selective retries.
Weights are unchanged; the method is not claimed as a new research invention.

The four primary fixes changed an older selected revision to the required newer
one: 8380 to 9169, 93 to 102, 8 to 11, and 4 to 20. In each case extraction kept
both competing records, so the second model call still had to select the winner.
Extraction itself was imperfect: three primary cases included extra irrelevant
records, although none omitted matching records and all final answers passed.
The confirmation extractions had no such errors. See [extraction-audit.json](extraction-audit.json).

## Settings and cost

- Ollama `0.35.0`, `qwen3.5:9b`, local metadata: 9.7B parameters, `Q4_K_M`.
- Model digest: `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`.
- Both workflows: `think=false`, `temperature=0`, `seed=20261015`,
  `num_ctx=4096`, `num_predict=512` per request; no forced JSON response format.
- Generator and scorer at experiment time:
  [`da6d174`](https://github.com/xMaxos/evidence-switch/commit/da6d174eb8981060eaf3bba4418712db4daf303c).
- Primary: seed 505, three groups per task and language; confirmation: seed 606,
  one group. English, German, and Spanish; update, missing, and distractor tasks.

| Primary test cost | Direct | Two-stage |
| --- | ---: | ---: |
| Model requests | 54 | 108 |
| Generated tokens | 2,047 | 7,758 |
| Total measured wall time | 52.4 s | 163.5 s |
| Median time per case | 1.10 s | 3.28 s |
| Median after subtracting model load time | 1.09 s | 3.26 s |

Confirmation medians were 0.89 s and 3.33 s. Hardware, caching, and request order
affect timings; these are measurements of this machine, not speed guarantees.
The two-stage method uses twice as many requests and a total possible output
budget of 1,024 rather than 512 tokens per case. It is not compute matched.

All 216 comparison requests completed with `done_reason=stop`, without output
truncation or thinking output. Three known development pilot cases used six
additional calls and are excluded from the 72 cases. API fees were **EUR 0**:
the experiment used an already installed local model, with no cloud calls,
downloads, or training. Ordinary computer electricity still applies.

## Experiment sequence and earlier attempts

The extraction instructions and settings in [frozen.json](frozen.json) were fixed
before evaluating seeds 505 and 606. Condition order alternated per case and was
reversed for confirmation. Every call started a new conversation. All 72 answer
keys were independently recomputed from the questions and supplied records.

Two earlier approaches were tried on different seeds and are disclosed to avoid
presenting only the positive result:

- A comparison checklist appended to the original instruction was developed on
  18 cases (seed 101), then tested on 54 new cases (seed 202). Strict correctness
  fell from **42/54 to 39/54**, and pairs from **16/27 to 14/27**: zero fixes,
  three new errors. It was not adopted. Its per-call output cap was 256 tokens.
- Built-in thinking was then tried. Three known pilot questions passed. The new
  seed-303 run was stopped after 12 saved thinking responses, three of which hit
  the frozen 2,048-token cap; 13 direct responses had been saved. Responses took
  about 15–32 seconds. This incomplete run provides no complete effect estimate.

The saved earlier [comparison](earlier-checklist-comparison.json) and
[abort record](earlier-thinking-aborted.json) support these summaries. Their full
raw experiments are not part of this public result bundle. The successful
two-stage comparison below includes complete raw evidence for both test sets.
Success after trying multiple approaches warrants further independent testing.

## Verify the published evidence without a model

From the repository root, with Python 3.10+:

```sh
python results/2026-10-15-qwen35-two-stage/verify.py
```

This sends no requests. It regenerates the cases, independently computes answer
keys, rescores all raw responses, checks extraction relevance and timing totals,
and replays saved extractions through the production workflow. All 216 model
message hashes must match the original measured requests. This demonstrates
prompt equivalence; it does not constitute a fresh 72-case model run of the
packaged command.

| Evidence | Primary | Confirmation |
| --- | --- | --- |
| Inputs and answer keys | [cases](holdout-cases.jsonl) | [cases](confirmation-cases.jsonl) |
| Raw direct answers | [baseline](holdout-baseline.jsonl) | [baseline](confirmation-baseline.jsonl) |
| Raw two-stage answers | [candidate](holdout-candidate.jsonl) | [candidate](confirmation-candidate.jsonl) |
| Raw extractions | [traces](holdout-traces.jsonl) | [traces](confirmation-traces.jsonl) |
| Per-request timing and prompt hashes | [timings](holdout-timings.jsonl) | [timings](confirmation-timings.jsonl) |
| Scores, fixes, and costs | [analysis](holdout-analysis.json) | [analysis](confirmation-analysis.json) |

These JSON/JSONL files are unchanged copies of the saved experiment evidence.
In timing records, the **outer** `condition` identifies the real workflow.
The inner call metadata says `baseline` in both workflows because the experiment
reused a helper that named its `think=false` configuration that way. It does not
mean that candidate calls used the baseline messages; their hashes are checked.
No hidden reasoning text is included.

## Packaged-command integration check

After integrating the workflow in version 0.2.0, both `run-ollama` strategies were
run on the six existing English fixtures in [examples/mini](../../examples/mini/).
Both scored **6/6**, with 6 direct and 12 two-stage requests. All requests stopped
normally, without thinking output. This verifies the packaged command, not an
additional improvement: these fixtures were already public and the direct
baseline made no mistakes. These six cases are excluded from the 72-case results.

The [integration-smoke evidence](integration-smoke/) includes raw predictions,
traces, settings, timing, and the comparison. It also used the installed local
model with zero API fees.

## Run a new local comparison

With the named model already installed and a local Ollama server running:

```sh
python -c "from pathlib import Path; Path('runs').mkdir(exist_ok=True)"
python -m evidenceswitch generate --out data/recheck-505 --seed 505 --groups 3
python -m evidenceswitch run-ollama --prompts data/recheck-505/prompts.jsonl --model qwen3.5:9b --strategy direct --out runs/recheck-direct.jsonl --trace runs/recheck-direct-trace.jsonl
python -m evidenceswitch run-ollama --prompts data/recheck-505/prompts.jsonl --model qwen3.5:9b --strategy two-stage --out runs/recheck-two-stage.jsonl --trace runs/recheck-two-stage-trace.jsonl
python -m evidenceswitch compare --cases data/recheck-505/cases.jsonl --baseline runs/recheck-direct.jsonl --candidate runs/recheck-two-stage.jsonl --out runs/recheck-comparison.json
```

The native runner defaults match the saved inference settings. It restricts the
endpoint to loopback and checks that the model is installed locally. It does not
download models. Choose new output paths for every run. These commands run the
conditions sequentially, unlike the historical alternating schedule; timings
are not directly comparable. Model versions and local inference can change
answers even with fixed seeds. To test generalization, use fresh seeds and
different models instead of tuning on the published cases.
