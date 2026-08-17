# Stage 5: multilingual embeddings

This stage converts the cleaned `CleanedTextUnit` JSONL from stage 4 into an aligned, normalized `float32` matrix.
It reads the corpus in batches and writes directly to a NumPy memmap, so memory use does not grow with corpus size.

## Command

Install the optional ML dependencies, then run:

```bash
uv sync --extra analysis
uv run python scripts/generate_embeddings.py \
  data/processed/cleaning/development-clean.jsonl \
  --config configs/embeddings.example.json \
  --output-dir data/processed/embeddings
```

Use `--limit 1000 --force` for a disposable trial. If a full run is interrupted after at least one completed batch,
repeat the same command with `--resume`. Resume is rejected when the input checksum, configuration, record limit, or
execution device differs. `--force` deliberately removes both completed and partial artifacts before starting over.

## Artifacts and alignment

```text
embeddings.npy
record-ids.jsonl
embedding-manifest.json
embedding-report.md
```

Row `i` of the matrix belongs to line `i` of `record-ids.jsonl` and line `i` of the cleaned JSONL (blank input lines
are ignored consistently). The manifest binds the matrix and IDs to the input SHA-256, model configuration, shape,
dtype, device, and creation time. Stage 6 verifies this contract before semantic deduplication.

Partial files are hidden in the output directory. A checkpoint is updated atomically only after the corresponding
matrix batch has been flushed. Raw texts are never copied into reports or manifests.

## Model and resources

The default model is `intfloat/multilingual-e5-large`; pin `model_revision` for fully repeatable production runs.
E5 receives the `query: ` prefix by default because these vectors represent individual texts for symmetric
similarity. Override `prompt_prefix` only after checking the selected model card and validating retrieval quality.

GPU is selected automatically when supported by sentence-transformers. Set `device` to `cpu` or `cuda` to require a
specific backend. Reduce `batch_size` after a CUDA out-of-memory error. The final matrix uses approximately
`records × dimensions × 4` bytes; 260,000 vectors of dimension 1,024 require about 1 GiB on disk.

## Quality gates

Every batch must have a stable two-dimensional shape, finite values, non-zero rows, and unit norms when normalization
is enabled. Duplicate record IDs, empty datasets, changed inputs during execution, and incompatible checkpoints fail
closed. Tests use an injected fake encoder and never download model weights.
