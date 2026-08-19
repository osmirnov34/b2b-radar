# Stage 6: semantic deduplication

This stage groups cleaned text units whose normalized embeddings have cosine similarity at or above a
model-specific threshold. It replaces the removed legacy blockwise all-pairs implementation with HNSW
candidate search, avoiding `O(n²)` similarity work on the full corpus.

## Structure

1. **Alignment contract.** An embedding manifest binds the cleaned-record SHA-256, embeddings SHA-256, row count,
   dimensions, model name, and normalization flag. Matching row counts alone are not accepted.
2. **Vector validation.** The stage requires a finite two-dimensional matrix with no zero vectors and normalizes rows
   before cosine comparison.
3. **ANN candidate generation.** Production uses deterministic single-threaded HNSW construction and query. The exact
   backend exists for unit tests and small threshold-calibration samples only.
4. **Role boundary.** `comment` and `reply` candidates are not merged by default, even when their text and vectors are
   identical.
5. **Threshold graph.** Candidate edges passing the cosine threshold form connected components. The earliest row is
   the deterministic representative.
6. **Audit samples.** Sample pairs contain indexes and cosine values, never raw text. Manual review joins indexes to
   local cleaned data.
7. **Artifacts.** Keep indexes, index-only groups, a manifest, and a safe Markdown report are written under ignored
   `data/` paths.
8. **Evaluation.** Tune thresholds and HNSW recall on development, validate once on validation, and do not select
   parameters using test.

## Standalone command

Stage 5 generates the matrix and manifest. Their checksums are required so a same-sized matrix cannot be silently
paired with the wrong records.

```bash
uv sync --extra analysis
uv run python scripts/semantic_deduplicate.py \
  data/processed/cleaning/development-clean.jsonl \
  data/processed/embeddings/embeddings.npy \
  --embedding-manifest data/processed/embeddings/embedding-manifest.json \
  --config configs/semantic-deduplication.example.json \
  --output-dir data/processed/semantic-deduplication
```

Outputs:

```text
keep-indices.json
semantic-groups.jsonl
semantic-deduplication-manifest.json
semantic-deduplication-report.md
```

## Calibration and limitations

HNSW is approximate. `ann_neighbors` bounds candidates per row; `ann_ef_search` controls search recall. Compare HNSW
groups against the exact backend on a manageable, labelled development subset before accepting parameters. Dense
duplicate groups may require a larger neighbour count. Transitive components can contain endpoints whose direct
similarity is below the threshold, so manual review must include whole groups, not only individual edges.

`block_size` remains in `DeduplicationConfig` only for backward compatibility with older configuration files and is
not used by the HNSW backend.
