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

## Min-cluster-size experiment

Use `notebooks/03_colab_clustering_grid.ipynb` to compare the fixed grid `[50, 100, 150, 250]` after a source run has
completed stage 8. Every variant uses the same checksum-bound development corpus and `clustering-reduced.npy`; only
`min_cluster_size` changes. The source run is never modified.

Grid checkpoints are stored under
`/content/drive/MyDrive/b2b-radar/ml-experiments/<run-id>/min-cluster-size-grid/`. Each variant owns a complete normal
clustering manifest, labels, probabilities, summary, and model. After a Colab disconnect, rerunning the grid verifies
their checksums and skips completed variants. A partial, tampered, or differently configured directory blocks rather
than overwriting evidence.

The final `clustering-grid-manifest.json` binds every result to the reduction, reduced matrix, corpus, complete base
HDBSCAN configuration, and variant manifest checksum. Its comparison includes cluster count, outlier share, cluster
sizes, mean membership probability, low-confidence share, dominant-cluster share, Relative Validity, DBCV when
configured, and warnings. These diagnostic labels do not replace stage 9 of the source pipeline automatically.

The same operation is available as a CLI:

```bash
python scripts/run_clustering_grid.py \
  /path/to/08-reduction/clustering-reduced.npy \
  --reduction-manifest /path/to/08-reduction/clustering-manifest.json \
  --corpus-manifest /path/to/07-corpus/corpus-manifest.json \
  --config configs/clustering-grid.example.json \
  --output-dir /path/outside/ml-runs/min-cluster-size-grid
```

Do not choose the production value from outlier share alone. Topic coherence, stability, source concentration, and
manual validation are separate decision criteria covered by later interpretability stages.

### Cross-variant cluster matching

After the grid completes, the same notebook matches every adjacent pair (`50→100`, `100→150`, `150→250`) by
intersecting their aligned corpus row indices. Numeric cluster IDs are local to a variant and are never compared as
identities. An overlap is material only when it meets both `minimum_shared_records` and
`minimum_overlap_share` for the source and target cluster; the notebook defaults to 5%.

For each material edge, `cluster-transitions.jsonl` records intersection size, Jaccard similarity, source retention,
target composition, split/merge status, and whether it is a primary link. A primary link requires a deterministic
mutual-best overlap: both clusters must select each other, with ID used only to resolve an exact tie. This keeps the
relationship one-to-one without adding a SciPy runtime dependency. Other material edges remain visible because they
are the evidence for splits and merges. Clusters without a material edge are recorded as `new` or `disappeared`,
which means “no material match under this threshold,” not necessarily that every record is novel or lost.

`cluster-matching-manifest.json` binds the transition file to the grid-manifest checksum and matching thresholds.
Existing output is checksum-validated and reused; partial or changed output blocks instead of being overwritten.
The notebook presents the complete transition table and a Sankey diagram, with mutual-best links highlighted.
