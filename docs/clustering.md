# Stage 9: HDBSCAN clustering

Stage 9 clusters only `clustering-reduced.npy` from stage 8. A visualization manifest is rejected even if its matrix
has a plausible shape. Parameter selection belongs to the development corpus; validation is reserved for stability
checks and test data must remain unopened.

## Command

```bash
uv sync --extra analysis
uv run python scripts/cluster_corpus.py \
  data/processed/umap/clustering-reduced.npy \
  --reduction-manifest data/processed/umap/clustering-manifest.json \
  --corpus-manifest data/processed/corpus/corpus-manifest.json \
  --config configs/hdbscan.example.json \
  --output-dir data/processed/clustering
```

Use `--limit 10000 --force` only for a disposable development trial. A limited labels artifact is aligned with the
corresponding corpus prefix and must not be treated as a complete clustering run.

## Artifacts

```text
cluster-labels.npy
cluster-probabilities.npy
cluster-summary.jsonl
hdbscan-model.pkl
clustering-manifest.json
clustering-report.md
```

Row `i` of labels and probabilities corresponds to row `i` of the final corpus. Label `-1` is an outlier. Other
labels are deterministically remapped to `0..N-1` by descending cluster size, then earliest corpus index. The manifest
stores the original-to-normalized mapping because the persisted HDBSCAN model still emits its original labels.

The summary contains only aggregate counts: size, corpus share, probability statistics, comment/reply counts,
language counts, number of videos, and minimum row index. It never contains comment text or source identifiers.

## Diagnostics and calibration

The manifest and report record cluster counts and sizes, outlier share, mean probability, low-confidence share,
dominant-cluster share, micro-cluster share, HDBSCAN relative validity, and optional DBCV. DBCV is disabled by default
because it can be expensive; set `dbcv_sample_size` only for a bounded development sample.

Warnings identify all-outlier results, excessive outliers, one dominant cluster, too many clusters near the configured
minimum size, and low membership confidence. They are diagnostics rather than automatic parameter changes. Compare a
small documented parameter grid on development, then perform one stability check on validation.

## Safety and reproducibility

The stage verifies the reduction matrix, reduction manifest, corpus manifest, corpus JSONL, and record-ID checksums.
Labels and probabilities are validated before publication. Outputs use temporary files and the manifest is published
last. Existing results require `--force`.

`hdbscan-model.pkl` is a local trusted artifact. This module writes it but provides no pickle loader. Never deserialize
a model from an untrusted source or before verifying its SHA-256 and origin.

Stages 10 and 11 consume normalized labels, probabilities, summary, and `clustering-manifest.json`.
