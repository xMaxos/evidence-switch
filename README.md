# EvidenceSwitch

**Small, reproducible regression checks for evidence-sensitive AI answers.**

When the evidence changes, does the answer change? When the evidence disappears,
does the model admit it? When an irrelevant record changes, does the answer stay
correct?

EvidenceSwitch generates paired, fictional record-reading tasks and scores the
answers without a model-based judge. Use it to catch regressions while changing a
model, system prompt, or inference setup. Python 3.10+, no runtime dependencies,
MIT licensed. Generation, scoring, comparison, and the demo work offline.

It does **not** train models or prove general intelligence. It makes one narrow
class of mistakes easier to reproduce and fix.

Version 0.3.0 adds optional **source-ID selection with an explicit answer format**.
On 180 local Llama 3.1 cases, it improved strict correctness from **112/180 to
167/180** versus the original two-stage workflow, with 56 fixes and one regression.
Median runtime fell 70.2% and generated output tokens fell 72.7%; input tokens
increased. Much of the accuracy gain was better format compliance.
[Read the Llama results, diagnostics, and raw evidence](results/2026-10-15-llama31/).
On a separate 36-case Qwen compatibility check, both two-stage methods scored
36/36; the new option used fewer output tokens, with more input tokens.

The original two-stage workflow remains available, and `direct` remains the
default. The earlier [72-case Qwen comparison](results/2026-10-15-qwen35-two-stage/)
and unsuccessful earlier attempts remain documented. These are narrow workflow
tests, not evidence of generally greater intelligence.

## Try it in a minute

From this repository's directory:

```sh
python -m evidenceswitch demo --out demo
```

Expected output includes:

```text
SIMULATED oracle: 100.0% pair accuracy
SIMULATED stale: 33.3% pair accuracy
These fixtures demonstrate scoring, not real model performance.
```

The oracle copies the answer key. The stale responder always copies the original
answer, even after evidence changes or disappears. Neither is a real AI model.
The demo saves the cases, prompts, predictions, and detailed reports for inspection.

Optional installation from this checkout:

```sh
python -m pip install .
evidenceswitch --help
```

This project is not currently published on PyPI. Use the repository's source.

## What is tested?

Each pair has two independently presented prompts with the same question.
The rules require an exact entity-and-field match, then the highest **numeric**
revision among matching records. A correct answer cites exactly that record.

| Task | Change in the second prompt | Expected behavior |
| --- | --- | --- |
| `update` | The matching evidence receives a new value and revision | Change the answer |
| `missing` | All matching evidence is replaced with irrelevant records | Return null and no citations |
| `distractor` | Irrelevant evidence changes | Keep the same answer and citation |

English, German, and Spanish instructions are included. Record field names stay
the same across languages. All entities and values are artificial, not facts
about real people. Record order, source identifiers, revisions, and values vary
with the seed. Case identifiers are bound to prompt contents. Paired cases are
shuffled and must be run in separate conversations.

## Generate and evaluate

```sh
python -m evidenceswitch generate --out data --seed 42 --groups 10
```

This produces **180 cases = 10 pairs × 3 tasks × 3 languages × 2 variants**:

- `data/prompts.jsonl`: only case IDs and model-facing messages.
- `data/cases.jsonl`: messages, pair metadata, and answer keys. Keep this out of model inputs.
- `data/manifest.json`: seed, generator version, and generation settings.

`--groups` is the number of pairs **per task per language**.
Use `--languages de` or `--languages en es` to select a subset.
Output directories and files are never overwritten; choose a new path for each run.
For a small inspectable fixture, see [examples/mini](examples/mini): six English
cases generated with seed 7 and one pair per task.

Collect one prediction per case in JSONL:

```json
{"id":"<case ID from prompts.jsonl>","response":{"answer":"<exact value>","citations":["<source ID>"]}}
```

`response` can also contain the model's raw response as a JSON-encoded string.
For absent evidence, the required response is `{"answer":null,"citations":[]}`.
Extra keys, duplicate keys/citations, code fences, and non-JSON output fail format
checks. Answers use exact string matching; source-ID array order is ignored.

```sh
python -m evidenceswitch score --cases data/cases.jsonl --responses responses.jsonl --out report.json
```

Missing predictions count as failures against the **entire** case set. Unknown or
duplicate prediction IDs and broken pairs are rejected. A partial `--limit` run
therefore shows partial coverage; it is not a score for a smaller test set.

## Optional model endpoint runner

If you already have a server exposing an OpenAI-compatible Chat Completions API:

```sh
python -m evidenceswitch run --prompts data/prompts.jsonl --base-url http://localhost:8000/v1 --model your-model --out responses.jsonl --limit 6
```

This command sends requests. Hosted endpoints may charge for them. Start with a
small limit; remove it explicitly when ready for the full set. No hosted model
evaluation was performed to produce the included demo.

For authentication, set your key in an environment variable and pass only its
name using `--api-key-env MODEL_API_KEY`. The runner sends messages, model,
`temperature=0`, and `stream=false`; it never sends case IDs, pair metadata, or
answer keys. Input files with extra fields are rejected before any request.

Requests are sequential, with no retries, redirects, or environment proxies.
HTTPS is required except on loopback hosts. Completed responses are kept if a
later request fails. A failed output is not automatically resumed; use a new
path. `--timeout 60` controls each request timeout. The endpoint must support
these Chat Completions parameters and a string `choices[0].message.content`;
other API styles and provider-specific reasoning options are not supported.
Model outputs are saved verbatim locally, so review them before sharing.

## Two-stage evidence selection

Both model runners accept these strategies. The two-stage modes support the
structured English, German, and Spanish prompts generated by EvidenceSwitch:

| Strategy | Selection and final answer | Calls per case |
| --- | --- | ---: |
| `direct` (default) | Answer the original prompt | 1 |
| `two-stage` | Copy full matching records, then answer | Up to 2 |
| `two-stage-ids` | Select source IDs, restore their original records, then answer with an explicit format reminder | Up to 2 |

The original `two-stage` workflow:

1. Ask the model to copy all records matching the requested entity and field,
   including both older and newer revisions.
2. Validate that the returned records are unchanged members of the input and
   have unique source identifiers. Code does not decide relevance or the answer.
3. Ask the same model to answer the original question from those records in a
   fresh conversation.

Invalid extraction produces a failed prediction, without repair or a second
call. The validator rejects invented or edited records, but it cannot detect an
omitted relevant record or an irrelevant copied record. That remains part of the
model's task. Valid extraction therefore does not guarantee a correct answer.
No answer keys enter either request, and model weights do not change.

The new `two-stage-ids` option asks for source IDs instead of copying whole
records. Code rejects unknown or duplicate IDs and restores the corresponding
original records in the model-selected order. It does not choose matching
records or compute the highest revision. The final instruction also explicitly
requires citation strings, with a format example containing placeholders.
This saves output work and addresses format mistakes; it is not training or an
automatic correction of the final answer. The production option includes both
changes; the plain-ID ablation in the result files does not.

Two-stage mode needs up to twice as many requests; hosted endpoints may charge
for both. Pass `--trace trace.jsonl` to save extraction responses and request
metadata alongside predictions. Inspect traces before sharing them.

## Run an existing local Ollama model

If Ollama and a model are already installed, this native runner keeps requests
on loopback and checks the model is local. It does not download models:

```sh
python -c "from pathlib import Path; Path('runs').mkdir(exist_ok=True)"
python -m evidenceswitch run-ollama --prompts examples/mini/prompts.jsonl --model llama3.1:8b --strategy two-stage-ids --out runs/local-ids.jsonl --trace runs/local-trace.jsonl
python -m evidenceswitch score --cases examples/mini/cases.jsonl --responses runs/local-ids.jsonl
```

The default endpoint is `http://127.0.0.1:11434`. Requests use `think=false`,
`temperature=0`, `seed=20261015`, `num_ctx=4096`, and `num_predict=512` per call,
matching the published experiment. Override `--seed`, `--num-ctx`,
`--num-predict`, `--timeout`, or `--base-url` when needed; the base URL must remain
loopback. The model must support these native Ollama options. Local inference
has no hosted API fee, but uses the computer's resources and electricity.

Use the same inputs with `--strategy direct` and new output paths for a baseline,
then `compare` as below. Use `--strategy two-stage` to compare against full-record
selection. Both published result bundles include raw responses, settings, costs,
limitations, and verification commands that work offline without a model:

```sh
python results/2026-10-15-qwen35-two-stage/verify.py
python results/2026-10-15-llama31/verify.py
```

## Interpret the report

| Metric | Meaning |
| --- | --- |
| `coverage` | Fraction of cases with a submitted prediction |
| `format_accuracy` | Fraction with the required JSON shape |
| `answer_accuracy` | Fraction with a valid shape and the exact expected answer |
| `citation_accuracy` | Fraction with a valid shape and exactly the expected source IDs |
| `strict_accuracy` | Fraction passing both answer and citation checks |
| `pair_accuracy` | Fraction of pairs with **both** cases strictly correct |
| `abstention_accuracy` | Fraction of unanswerable cases returning null in a valid shape |
| `false_abstention_rate` | Fraction of answerable cases returning null in a valid shape |

Each metric includes missing predictions in its denominator. Abstention checks
look at the answer; citation correctness is measured separately. Rates with no
applicable cases are `null`. Reports also include per-task and per-language
summaries and per-case failure flags. Citation correctness checks labels, not
whether the model causally relied on the cited text.

## Catch regressions in CI

Use the **same generated cases** for both runs:

```sh
python -m evidenceswitch compare --cases data/cases.jsonl --baseline baseline.jsonl --candidate candidate.jsonl --out comparison.json --fail-on-regression
```

The report lists individual fixes and regressions, even if the overall score is
unchanged. Exit code `1` indicates at least one strict-correct case regressed.
For a score threshold, use `score --min-pair-accuracy 0.9`. Exit codes: `0` success,
`1` evaluation gate failed, `2` invalid input, file, or endpoint error.

## Limits and related work

These are small synthetic structured-record tasks. They do not cover open-ended
factuality, real retrieval quality, long-context reasoning, multi-turn learning,
prompt injection, or general multilingual competence. A parser can solve them;
that is useful for checking the harness, not evidence of intelligence. Template
shortcuts can remain. New seeds change fixtures, not the underlying task family.
Do not tune on one seed and treat its score as unseen-test performance. Record
the seed, generator version, model version, and inference settings when sharing
results. Scores on different generator versions are not directly comparable.

Context-sensitive evaluation already has substantial prior work:

- [FaithEval](https://github.com/SalesforceAIResearch/FaithEval) evaluates unanswerable, inconsistent, and counterfactual contexts.
- [PFaithBench](https://arxiv.org/abs/2610.07894) studies paired context-sensitive faithfulness.
- [ContextCite](https://github.com/MadryLab/context-cite) investigates attribution of generated statements to context.

EvidenceSwitch is a compact engineering tool, not a replacement for these
benchmarks and not a claim of a new research method. No external datasets or
implementation code are bundled.

## Contribute

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
