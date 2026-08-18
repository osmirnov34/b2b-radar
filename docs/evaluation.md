# Stage 12: topic quality evaluation

Stage 12 evaluates the fixed outputs of stages 7–11. It does not tune labels. The result combines embedding geometry,
bootstrap stability, topic representation checks, before/after reassignment comparison, and optional human review.

## Run

```bash
python scripts/evaluate_topics.py \
  data/processed/corpus/final-embeddings.npy \
  data/processed/clustering/cluster-labels.npy \
  data/processed/outlier-reassignment/final-cluster-labels.npy \
  data/processed/outlier-reassignment/final-cluster-confidence.npy \
  --corpus data/processed/corpus/final-corpus.jsonl \
  --corpus-manifest data/processed/corpus/corpus-manifest.json \
  --clustering-manifest data/processed/clustering/clustering-manifest.json \
  --topic-manifest data/processed/topics/topic-representation-manifest.json \
  --reassignment-manifest data/processed/outlier-reassignment/outlier-reassignment-manifest.json
```

The bootstrap reuses the exact UMAP and HDBSCAN settings recorded by their manifests. Set `bootstrap_runs` to zero for
a fast contract/manual-review preparation run. The deterministic geometry and bootstrap samples use only development
data at the current pipeline stage.

## Human review and verdict

`manual-review-sample.jsonl` contains comment text and therefore must stay below ignored `data/`. Copy
`manual-review-template.json` to a JSONL file, make one annotation per sampled `record_index`, and pass it with
`--manual-annotations`. Review topic fit, clarity, business relevance, reassignment correctness, sensitive data, and
possible merge/split cases.

No run can receive final `pass` until annotations are supplied and `validation_completed` is explicitly enabled in a
frozen evaluation config. Test data is not used for tuning. A failed threshold produces `fail`; missing validation or
manual review produces the auditable preliminary status `pass_with_warnings`.

## Outputs

- `evaluation-metrics.json`: aggregate machine and manual metrics;
- `topic-evaluation.jsonl`: topic-level counts and representative similarity, without keywords;
- `cluster-matching.jsonl`: Hungarian bootstrap matches and split/merge indicators;
- `manual-review-sample.jsonl`: sensitive local review data;
- `evaluation-report.md`: text-free summary safe to share;
- `evaluation-manifest.json`: hashes, configuration, warnings, and artifact lineage.

The manifest is published last and existing output is protected unless `--force` is supplied.
