# Decisions

2026-11-14, working tree based on `454185c`: keep the existing exact cache unchanged and add an opt-in guarded second tier. Scope and guard equality precede cosine similarity because the supplied adversarial pairs show that high similarity does not prove equivalent meaning. Response JSON lives in separate Chroma documents so semantic retrieval does not depend on a second Redis lookup. This makes local disk and backups sensitive.

Use the maximum of the per-dataset guarded dev thresholds as the shared runtime threshold. Keep per-dataset and shared-threshold test scores separate in `bench/results/threshold.json`. The held-out precision target is not met; enabling remains an explicit operator choice. Retain version-1 exact key compatibility and include changed versions in the request namespace.
