# Changelog

Repository versions; these entries are not PyPI releases. Each linked report
identifies its own model, baseline and test seed. Scores across different test
sets are not a version leaderboard.

## 0.5.0 — Numeric revision selection

- Add optional `two-stage-latest`: execute the model's exact query, select the
  unique highest integer revision, then ask the model for its answer.
- Reject ambiguous highest revisions without guessing; preserve older modes
  and the `direct` default.
- Reproduce and test the two known Llama revision errors separately from fresh
  cases on both models. [Evidence](results/2026-10-15-latest-revision/).
- Shorten the project overview; separate usage and methodology; index all
  experiments and verify them with one offline command.

## 0.4.0 — Exact retrieval

- Add `two-stage-retrieval`: the model writes an entity/field query and code
  executes exact matching. The model still chooses the revision and answer.
- Publish fresh content checks on Llama and Qwen, plus rejected development
  attempts. [Evidence](results/2026-10-15-exact-retrieval/).

## 0.3.0 — Source IDs and answer format

- Add `two-stage-ids` with ID validation and a final answer-format reminder.
- Publish the Llama transfer test, a separate format diagnostic and Qwen
  compatibility evidence. [Evidence](results/2026-10-15-llama31/).

## 0.2.0 — Two model calls

- Add full-record selection followed by answering, with native local Ollama
  execution and traces. [Qwen evidence](results/2026-10-15-qwen35-two-stage/).

## 0.1.0 — Offline regression checks

- Generate fictional paired tasks in English, German and Spanish.
- Score exact values and citations, compare fixes/regressions, and demonstrate
  scoring with clearly labeled simulated responders.
