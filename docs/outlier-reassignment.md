# Stage 11: conservative outlier reassignment

Stage 11 creates new final labels without editing HDBSCAN artifacts. An outlier is assigned only when its normalized
multilingual embedding is sufficiently close to one reliable topic centroid and sufficiently far from the runner-up.
Ambiguous records remain `-1`.

## Command

```bash
uv run python scripts/reassign_outliers.py \
  data/processed/corpus/final-embeddings.npy \
  data/processed/clustering/cluster-labels.npy \
  --probabilities data/processed/clustering/cluster-probabilities.npy \
  --corpus data/processed/corpus/final-corpus.jsonl \
  --corpus-manifest data/processed/corpus/corpus-manifest.json \
  --clustering-manifest data/processed/clustering/clustering-manifest.json \
  --topic-manifest data/processed/topics/topic-representation-manifest.json \
  --config configs/outlier-reassignment.example.json \
  --output-dir data/processed/outlier-reassignment
```

The topic representation must cover every normalized HDBSCAN topic. A stage 10 `--limit-topics` trial is rejected.
Use `enabled: false` to produce an audited pass-through result, and `--force` to replace completed outputs.

## Decision policy

Centroids use only original cluster members above `centroid_member_minimum_probability`. A topic is eligible only when
its original size, mean HDBSCAN probability, number of centroid members, centroid cohesion, representation, and
separation from other centroids pass configured gates.

For two or more eligible topics, a candidate must satisfy:

```text
best cosine similarity >= similarity_threshold
best similarity - second similarity >= margin_threshold
```

With one eligible topic, the stricter `single_topic_similarity_threshold` applies. Accepted candidates are ordered by
similarity, margin, and corpus index before per-topic expansion and global reassignment limits are applied. No random
assignment or forced distribution exists.

All calculations use original multilingual embeddings. UMAP coordinates and c-TF-IDF weights are not used for
semantic reassignment.

## Artifacts

```text
final-cluster-labels.npy
final-cluster-confidence.npy
outlier-decisions.jsonl
final-cluster-summary.jsonl
outlier-reassignment-manifest.json
outlier-reassignment-report.md
```

Original HDBSCAN members retain their labels and probability confidence. Reassigned rows receive cosine similarity as
confidence. The manifest records both confidence sources explicitly; they must not be interpreted as the same
calibrated probability.

Every original outlier receives an index-only decision with its two closest eligible topics, similarities, margin,
final label, decision, and rejection reason. Summaries distinguish original members from reassigned rows. No comment,
author, video, or query value is written to decisions or reports.

## Calibration

Calibrate similarity, margin, centroid, and growth thresholds on development only. Review accepted and rejected index
samples locally, then perform a fixed stability check on validation. Do not tune against test data. High reassignment
share, low margins, excluded centroids, or one topic receiving most accepted outliers are warnings to tighten the
policy rather than reasons to distribute the remaining records automatically.
