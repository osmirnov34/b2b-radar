# Feedback ML adaptation

Stage 9 joins an existing checksum-verified ML run to the canonical dental feedback artifact. The boundary lives in
`src/operations`: neither the source-neutral analytics contracts nor the reusable ML package depends on the other.

The output contains feedback and organization identifiers, rating, date, topic, confidence, and reassignment status.
It deliberately excludes private review text. Records removed by development splitting, cleaning, or deduplication are
reported as unmatched feedback; every record retained in the final ML corpus must match exactly one canonical record.

```bash
uv run python scripts/adapt_feedback_ml_results.py \
  data/analytics/yandex-feedback.jsonl \
  data/analytics/yandex-feedback.manifest.json \
  data/ml-runs/<run-id>/07-corpus/corpus-manifest.json \
  data/ml-runs/<run-id>/11-reassignment/reassignment-manifest.json \
  data/analytics/yandex-feedback-topics.jsonl
```

The command never overwrites its JSONL output or the adjacent `.manifest.json`. It validates source checksums,
corpus/array lengths, reassignment decisions, topic references, and identifier uniqueness before publication.
