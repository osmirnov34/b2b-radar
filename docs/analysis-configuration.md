# Topic-analysis configuration

Runtime parameters are represented by the immutable Pydantic models in `src/analysis/config.py`. The same models are
intended for the CLI, notebooks, tests, and future analysis services. Unknown fields are rejected so misspelled
experiment parameters cannot be silently ignored.

## Configuration groups

### Cleaning

| Field | Default | Constraint |
|---|---:|---|
| `min_length` | `20` | Integer from 0 to 10,000 |
| `spam_filter` | `false` | Apply the shared pipeline noise gate |

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

Thresholds are embedding-model-specific and should be checked against sample duplicate pairs before a production run.

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

from src.analysis import AnalysisConfig, EmbeddingConfig

config = AnalysisConfig(
    input_path=Path("data/comments.jsonl"),
    embedding=EmbeddingConfig(model_name="deepvk/USER-bge-m3", batch_size=16),
)
```

Configuration can be saved and loaded without losing path or nested-model values:

```python
from src.analysis import load_analysis_config, save_analysis_config

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
