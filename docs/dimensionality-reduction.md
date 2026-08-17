# Stage 8: UMAP dimensionality reduction

Stage 8 creates two independent projections from the aligned stage 7 embeddings:

- `clustering-reduced.npy` is a configurable 5-dimensional space used by HDBSCAN;
- `visualization-2d.npy` is a 2-dimensional space used only for plots and manual diagnostics.

The visualization projection must never replace the clustering projection as model input.

## Command

```bash
uv sync --extra analysis
uv run python scripts/reduce_dimensions.py \
  data/processed/corpus/final-embeddings.npy \
  --corpus-manifest data/processed/corpus/corpus-manifest.json \
  --config configs/umap.example.json \
  --output-dir data/processed/umap \
  --mode both
```

Use `--mode clustering` or `--mode visualization` for one space. `--limit 10000 --force` creates a disposable
prefix-based trial; its manifest records both the full input count and reduced output count. Full clustering must not
consume a limited trial artifact.

## Scale strategy

Input vectors are memory-mapped. By default, UMAP is fitted on a deterministic sample of at most 100,000 vectors,
then the corpus is transformed and written to a `float32` memmap in configurable blocks. Set
`training_sample_size` to `null` to train on all vectors when RAM and runtime allow it. The training indexes are
sorted before persistence, while transformation always preserves original corpus order.

`random_seed` and one UMAP worker are enforced together for reproducibility. Increasing `threads` is rejected because
parallel stochastic optimization does not provide the same deterministic guarantee.

## Artifacts

```text
clustering-reduced.npy
clustering-model.pkl
clustering-training-indices.json
clustering-manifest.json
clustering-report.md
visualization-2d.npy
visualization-model.pkl
visualization-training-indices.json
visualization-manifest.json
visualization-report.md
```

Each mode has its own manifest and completion marker. The manifest binds the corpus, reduced matrix, training
indexes, model checksum, configuration, effective neighbour count, library version, and quality diagnostics.

Model files are local pickle artifacts. This stage only writes them. Never deserialize a model received from another
machine or an untrusted location; verify its SHA-256 and provenance first.

## Quality gates

The stage rejects a changed corpus, misaligned IDs, incompatible shapes, non-finite input, zero input vectors,
incorrect reducer output, non-finite coordinates, and zero-variance dimensions. Reports include coordinate variance,
duplicate-coordinate share, and trustworthiness on a bounded prefix sample. They contain no comment text or source
metadata.

Stage 9 must consume `clustering-reduced.npy` together with `clustering-manifest.json`.
