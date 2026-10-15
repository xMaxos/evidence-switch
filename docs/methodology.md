# Test methodology

[Project overview](../README.md) · [Usage](usage.md) · [Results](../results/README.md)

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

## Content scoring in experiments

The CLI metrics above require the strict final JSON shape. Recent experiment
reports additionally measure exact answer values independent of citation shape
and extra keys, and grounded content with normalized citation representations.
The frozen plan in each bundle defines its scoring rule. No model response is
repaired during inference. A missing/unparseable answer still fails content
scoring; an improved count alone does not prove improved reasoning.

Known failing cases are reused **regression fixtures**, not new efficacy tests.
Development cases guide changes. Main tests use a new frozen seed and unchanged
candidate code, with both conditions applied to the same cases. Both models
share a test set; 180 questions per model means 180 distinct questions and 360
model/task evaluations. Do not combine scores from different seeds into a
single version ranking or treat multiple variants of a pair as independent.

## What deterministic tools change

`two-stage-retrieval` delegates exact entity/field matching to code.
`two-stage-latest` additionally delegates numeric revision selection. In the
latter, the model's remaining work is query extraction and copying/citing the
retrieved source, or abstaining on an empty result. High scores on these tasks
therefore measure the combined workflow, not the model's independent ability
to choose evidence or compare revisions. Model weights never change.
