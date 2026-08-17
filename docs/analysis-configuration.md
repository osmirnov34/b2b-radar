# Topic-analysis configuration

Runtime parameters are represented by the immutable Pydantic models in `src/ml/config.py`. The same models are
intended for the CLI, notebooks, tests, and future analysis services. Unknown fields are rejected so misspelled
experiment parameters cannot be silently ignored.

## Configuration groups

### Cleaning

| Field | Default | Constraint |
|---|---:|---|
| `schema_version` | `1` | Cleaning contract version |
| `min_length` | `20` | Integer from 0 to 10,000 |
| `max_length` | `null` | Null or at least 1 and not below `min_length` |
| `unicode_normalization` | `NFKC` | `NFC` or `NFKC` |
| `strip_html` | `true` | Decode entities and remove tags |
| `url_handling` | `token` | `keep`, `token`, or `remove` |
| `acknowledgement_filter` | `true` | Remove only short acknowledgement-only text |
| `spam_filter` | `false` | Apply the shared pipeline noise gate |
| `repeated_char_filter` | `true` | Remove runs of six or more repeated characters |
| `uppercase_filter` | `false` | Remove mostly-uppercase text |
| `detect_language` | `true` | Attach a Lingua label |
| `allowed_languages` | `[]` | Empty means detect but do not filter |
| `exact_deduplication` | `true` | Keep the first normalized representative |

The standalone dataset workflow uses every field and has its own complete example at
`configs/dataset-cleaning.example.json`. The older topic-mining CLI continues to use its established `min_length` and
`spam_filter` behavior until it is migrated to consume `development-clean.jsonl` directly.

### Embeddings

| Field | Default | Constraint |
|---|---:|---|
| `model_name` | `intfloat/multilingual-e5-large` | Non-blank string |
| `threads` | `4` | At least 1 |
| `batch_size` | `64` | At least 1 |
| `max_seq_length` | `512` | At least 8 |

The CLI keeps its historical `--batch-size 0` shorthand and translates it to the internal value `64`.

### Semantic deduplication

| Field | Default | Constraint |
|---|---:|---|
| `enabled` | `true` | Enable cosine near-duplicate removal |
| `threshold` | `0.95` | Inclusive range 0..1 |
| `block_size` | `2048` | At least 1 |
| `sample_pairs` | `8` | Non-negative |
| `backend` | `hnsw` | `hnsw` for production or `exhaustive` for small calibration sets |
| `exhaustive_max_records` | `50000` | Hard safety limit for quadratic calibration runs |
| `ann_neighbors` | `64` | Candidate neighbours per vector, at least 2 |
| `ann_ef_construction` | `200` | HNSW construction quality, at least 2 |
| `ann_ef_search` | `200` | HNSW query quality, at least 2 |
| `ann_m` | `16` | HNSW graph degree, at least 2 |
| `random_seed` | `42` | Deterministic HNSW seed |
| `separate_text_kinds` | `true` | Do not merge comments with replies |

Thresholds are embedding-model-specific and should be checked against sample duplicate pairs before a production run.
The structural workflow and ANN limitations are documented in
[`semantic-deduplication.md`](semantic-deduplication.md).

### Clustering

| Field | Default | Constraint |
|---|---:|---|
| `min_topic_size` | `250` | At least 2 |
| `reduce_outliers` | `true` | Enable thresholded outlier reassignment |
| `reduce_outliers_threshold` | `0.9` | Inclusive range 0..1 |
| `random_seed` | `42` | Integer |
| `top_n` | `50` | Non-negative |

### Run scope

`input_path` is required. `output_dir` defaults to `docs/analysis-output`. Internal `limit` and `sample_size` are either
positive integers or null. The CLI preserves `--limit 0` and `--sample 0` as shorthands for null.

Processing order is:

1. Load all input records.
2. Apply reproducible random sampling when `sample_size` is set.
3. Clean and remove exact duplicates.
4. Apply `limit` to the cleaned records.
5. Build embeddings, remove semantic duplicates, and cluster.

## Python and notebook usage

```python
from pathlib import Path

from src.ml import AnalysisConfig, EmbeddingConfig

config = AnalysisConfig(
    input_path=Path("data/comments.jsonl"),
    embedding=EmbeddingConfig(model_name="deepvk/USER-bge-m3", batch_size=16),
)
```

Configuration can be saved and loaded without losing path or nested-model values:

```python
from src.ml import load_analysis_config, save_analysis_config

save_analysis_config(config, Path("analysis.json"))
restored = load_analysis_config(Path("analysis.json"))
```

The repository example is `configs/topic-analysis.example.json`. File existence is deliberately checked by the future
analysis pipeline rather than by the model, allowing notebooks and tests to prepare configurations before data exists.

## CLI compatibility

Existing flags remain supported. CLI-only sentinel values are normalized before execution:

```text
--batch-size 0       -> embedding.batch_size = 64
--limit 0            -> limit = null
--sample 0           -> sample_size = null
--no-near-dup        -> deduplication.enabled = false
--no-reduce-outliers -> clustering.reduce_outliers = false
```

Invalid values are reported by argparse without a Python traceback. The CLI currently remains the authoritative way to
execute a full run; JSON configuration loading is available for notebooks and programmatic experimentation.
