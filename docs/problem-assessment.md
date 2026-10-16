# Hybrid problem assessment

Stage 10 produces conservative, multi-label candidates from canonical dental feedback. A category is emitted only
when its versioned literal term is present in the original text. Rating and explicit positive/negative language define
the stance. A matching topic from stage 9 can increase category confidence, but cannot create a category by itself.
When positive and negative markers overlap, the longer phrase wins: for example, `не больно` is positive and
`не рекомендую` is negative. Non-overlapping markers remain `mixed`.

Every result remains `unreviewed`. Evidence is persisted only as character offsets into restricted source text; the
assessment artifact contains no quotes or full review text. These candidates describe customer perception and must
not be presented as proven medical error or measured financial loss.

```bash
uv run python scripts/assess_feedback_problems.py \
  data/analytics/yandex-feedback.jsonl \
  data/analytics/yandex-feedback.manifest.json \
  data/analytics/yandex-feedback-problems.jsonl \
  --topic-assignments data/analytics/yandex-feedback-topics.jsonl \
  --topic-manifest data/analytics/yandex-feedback-topics.jsonl.manifest.json
```

The ML arguments may be omitted for a rules-only baseline. The output and adjacent manifest are never overwritten;
both the canonical feedback and optional topic assignments must match their SHA-256-bound manifests.
