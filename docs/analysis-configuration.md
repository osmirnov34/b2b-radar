# ML pipeline configuration

The supported ML workflow uses `configs/pipeline.example.json` as its orchestration contract. Each model or data stage
owns a separate immutable Pydantic configuration with unknown fields forbidden. The removed monolithic topic-analysis
configuration is not a supported execution path.

## Configuration ownership

| Stage | Example | Owning model |
|---|---|---|
| Cleaning | `dataset-cleaning.example.json` | `CleaningConfig` |
| Embeddings | `embeddings.example.json` | `EmbeddingConfig` |
| Semantic deduplication | `semantic-deduplication.example.json` | `DeduplicationConfig` |
| UMAP | `umap.example.json` | `UMAPConfig` |
| HDBSCAN | `hdbscan.example.json` | `HDBSCANConfig` |
| Topic representation | `topic-representation.example.json` | `TopicRepresentationConfig` |
| Outlier reassignment | `outlier-reassignment.example.json` | `OutlierReassignmentConfig` |
| Evaluation | `evaluation.example.json` | `EvaluationConfig` |
| Export | `export.example.json` | `ExportConfig` |

`PipelineConfig` references those files and adds the source dataset, run directory, run identifier, Python executable,
disk-space threshold, expected record count, and optional manual annotations. Stage configurations are loaded and
validated by their owning model during dry-run before any real process starts.

## Supported entry point

```bash
uv run python scripts/run_ml_pipeline.py dry-run configs/pipeline.example.json
uv run python scripts/run_ml_pipeline.py smoke-run configs/pipeline.example.json --records 2000
uv run python scripts/run_ml_pipeline.py run configs/pipeline.example.json
```

The notebook calls the same `src.operations` API. It is an exploration and control surface, not a second pipeline
implementation. Complete execution, resume, restart, manual-review, publication, and rollback behavior is documented
in `ml-operations.md`.

## Reproducibility policy

- Copy example files to environment-specific configuration before a production run.
- Pin the embedding model revision.
- Keep deterministic seeds and single-threaded UMAP.
- Do not tune parameters against the test split.
- Set `validation_completed: true` only after manual validation.
- Require a final passing evaluation for production export.
- Preserve configuration snapshots and checksums stored in each run directory.
