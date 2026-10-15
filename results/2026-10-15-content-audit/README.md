# Unsuccessful development attempt: per-source boolean audit

This attempt was rejected before any new holdout test. Both models received the
same 18 development cases (seed 1107). The candidate asked for an ordered list
of all eight source IDs with separate entity-match and field-match booleans.
Code required complete coverage and used the model's two booleans to select
records. The final answer instructions were unchanged.

| Model | Existing ID selection: correct values | Audit: correct values |
| --- | ---: | ---: |
| Llama 3.1 8B | 15/18 | 1/18 |
| Qwen 3.5 9B | 18/18 | 12/18 |

Llama omitted sources and sometimes emitted entity strings instead of source
IDs. Incomplete or malformed audits failed, without repair or a second call.
The method is not exposed by the production CLI. All raw responses, the plan,
scripts and analyses are preserved here. No paid model service was used.

These development cases were reused for subsequent revisions. Their later
scores are tuning evidence, not independent confirmation. The final
[exact-retrieval experiment](../2026-10-15-exact-retrieval/) uses a new holdout seed.
