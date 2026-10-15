# Unsuccessful development attempt: relevance-only source view

This second attempt removed values and revisions from the selection prompt,
showing every record's source ID, entity and field. Selected IDs still resolved
to the unchanged full records. Both the selection system instructions and the
final answer format were identical to the existing ID-selection workflow.

On the same 18 Llama development cases (seed 1107), both baseline and candidate
answered 15/18 correctly. There was no aggregate content gain. The method was
not pursued on Qwen or a holdout and is not a production CLI option.

Raw responses and the frozen development plan are retained. This was a tuning
attempt after the [boolean audit](../2026-10-15-content-audit/) failed; it is not
an independent test. The later [exact-retrieval experiment](../2026-10-15-exact-retrieval/)
has its own fresh holdout. All model requests were local, with zero API fees.
