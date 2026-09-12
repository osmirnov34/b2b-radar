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

Use `notebooks/03_colab_clustering_grid.ipynb` to compare the fixed grid `[2, 3, 5, 10]` after a source run has
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

### Parameter stability trajectories

The stability step follows only mutual-primary links from the matching checkpoint. It never forces a path through a
secondary split/merge edge. Each trajectory retains its local `(min_cluster_size, cluster_id)` nodes and reports grid
coverage, transitions survived, mean/minimum Jaccard, mean/minimum source retention, and whether any followed edge
was involved in a split or merge.

The default levels are deliberately explainable:

- `stable`: covers the complete grid, has minimum Jaccard at least 0.50 and retention at least 0.70, with no
  split/merge ambiguity;
- `moderate`: covers at least half the grid, has minimum Jaccard at least 0.25 and retention at least 0.50;
- `fragile`: has a primary continuation but misses the moderate thresholds;
- `unmatched`: has no mutual-primary transition.

Thresholds belong to `ClusterStabilityConfig` and are persisted with the result. The level is a readable summary,
not a replacement for the component metrics. A split/merge trajectory may be moderate but cannot be `stable` under
the default policy. A cluster that is `unmatched` under the material-overlap threshold is not automatically a bad
topic; it requires manual inspection.

`cluster-stability.jsonl` and `cluster-stability-manifest.json` are stored beside the grid and matching outputs. They
are checksum-bound to the matching manifest and transition file, safely reused, and never written into the source
ML run. The Colab notebook shows the full trajectory table, stability-level counts per grid variant, and size paths
for the largest trajectories.

### Empirical cluster hierarchy

The hierarchy step treats the lower `min_cluster_size` cluster as a detailed child and the adjacent higher value as
a coarser parent. Every node keeps its local `(min_cluster_size, cluster_id)`, record count, stability trajectory,
and stability level. Equal numeric IDs at different levels are still unrelated unless a verified overlap edge joins
them.

Mutual-primary overlaps form the navigable parent paths. All other material overlaps remain as secondary DAG edges
instead of being discarded to manufacture a strict tree. For every edge the output preserves child containment,
parent composition, Jaccard, overlap size, and split/merge status. `root` means that a node has no coarser primary
parent; `leaf` means it has no more detailed primary child.

Independent HDBSCAN runs are not guaranteed to be nested. An edge is therefore marked `nesting_violation` when its
child containment is below `ClusterHierarchyConfig.minimum_parent_containment` (0.80 by default). This is a
diagnostic threshold, not a hidden relabelling rule. The resulting structure is explicitly an empirical DAG, not an
HDBSCAN condensed tree and not a semantic taxonomy.

The files `cluster-hierarchy-nodes.jsonl`, `cluster-hierarchy-edges.jsonl`, and
`cluster-hierarchy-manifest.json` are checksum-bound to both the stability and matching checkpoints. Partial,
modified, or differently configured output blocks reuse. The Colab notebook displays both node and edge tables;
its icicle chart includes primary parent paths only and uses equal node weights because independent clusters are not
strictly nested. Actual record counts remain in hover and tables; secondary overlaps and nesting violations remain
visible in the edge table.
