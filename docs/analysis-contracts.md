# Analysis data contracts

The offline topic-analysis workflow exchanges three versioned formats. Their executable Pydantic definitions live in
`src/analysis/schemas.py`; fixtures in `tests/fixtures/` are canonical examples.

Readers ignore unknown fields for forward compatibility. Producers must preserve the required fields and their
meaning. Text is UTF-8 in every format.

## Comments export (`comments.jsonl`)

One JSON object per line. `comment_text` is the only required field. Empty text is valid at the transport boundary and
may be removed by the cleaning stage. Missing or null optional provenance is normalized to an empty string; counters
default to zero.

| Field | Type | Meaning |
|---|---|---|
| `comment_text` | string | Original top-level comment text |
| `comment_id` | string | Platform comment identifier |
| `comment_author` | string | Display name supplied by the platform |
| `comment_published_at` | ISO 8601 datetime or null | Comment publication time |
| `comment_like_count` | non-negative integer | Comment likes |
| `comment_total_reply_count` | non-negative integer | Total replies reported by YouTube |
| `video_id` | string | YouTube video identifier |
| `video_title` | string | Video title at export time |
| `video_channel` | string | Channel title at export time |
| `video_url` | string | Source video URL |
| `search_query` | string | Query that first discovered the source |

## Cluster output (`clusters.jsonl`)

One non-outlier topic per line. This is the file accepted by the `/analysis` upload page.

| Field | Type | Constraint |
|---|---|---|
| `topic_id` | non-negative integer | BERTopic topic identifier; outlier `-1` is excluded |
| `n_comments` | non-negative integer | Must equal the length of `comments` |
| `n_authors` | non-negative integer | Distinct authors represented by the producer |
| `n_channels` | non-negative integer | Distinct channels represented by the producer |
| `keywords` | array of strings | Ordered topic keywords |
| `comments` | array of objects | Self-contained text and provenance records |

Each comment contains required `text` plus optional `author`, `channel`, `query`, `video_id`, `video_title`, and
`video_url` strings. Keeping provenance inside the record lets the web viewer render a cluster without joins.

## Run metadata (`run_meta.json`)

A single JSON object stored beside `clusters.jsonl`. `schema_version` is currently `1`; legacy files without this field
are interpreted as version 1.

The contract records the model and thresholds, processing counts, topic/outlier totals, and an ISO 8601 `created_at`.
Processing counts must satisfy:

```text
n_after_dedup <= n_after_clean <= n_input
n_outliers <= n_outliers_before_reduction
```

Thresholds are either null (stage disabled) or in the inclusive range `0..1`. `min_topic_size` is at least 2 and all
counts are non-negative.

## Compatibility policy

- Additive fields do not require a schema-version increment.
- Renaming/removing fields or changing their meaning requires a new schema version and a migration path.
- The CLI producer, Pydantic contracts, fixtures, and `/analysis` parser are covered by contract tests and must change
  together.
