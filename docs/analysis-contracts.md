# Analysis data contracts

The maintained pipeline accepts a versioned comments export, creates stage-specific manifests, and publishes the
strict contracts described in `export.md`. This document also records the compatibility-only `clusters.jsonl` and
`run_meta.json` formats still accepted by the historical `/analysis` upload page. Their executable Pydantic definitions
live in `src/ml/schemas.py`; fixtures in `tests/fixtures/` are canonical reader examples.

Readers ignore unknown fields for forward compatibility. Producers must preserve the required fields and their
meaning. Text is UTF-8 in every format.

## Yandex Maps reviews source (`yandex-reviews.jsonl`)

The source adapter accepts one strict JSON object per line. It rejects unreviewed extra fields, invalid identifiers,
non-calendar dates, coerced ratings, non-Yandex URLs, and organization URLs that disagree with `org_id`. Validation
diagnostics contain line numbers, field names and types, and content hashes, but never review or author values.

| Field | Type | Constraint |
|---|---|---|
| `review_id` | string | Exactly 16 lowercase hexadecimal characters |
| `review_text` | string | Original text preserved without whitespace normalization |
| `rating` | integer | Strict integer in `1..5` |
| `author` | string or null | Null normalizes to an empty restricted-source value |
| `published_at` | `YYYY-MM-DD` | Calendar date without time-of-day |
| `org_id` | string | Non-empty decimal Yandex organization identifier |
| `org_name` | string | Non-empty display name |
| `org_url` | string | HTTPS `yandex.ru` URL containing the same `org_id` |
| `search_query` | string or null | Null normalizes to an empty string |

`YandexReview` is a source contract, not the common ML record. Conversion to the source-neutral feedback contract is
performed separately so source semantics are not disguised as historical `video_*` fields.

## Canonical analytical feedback (`FeedbackRecord`)

Every supported adapter converts its verified source record to the immutable `FeedbackRecord` before analytical
storage or metric calculation. The canonical record uses business-neutral names and therefore does not make
PostgreSQL, reports, or BI views depend on Yandex-specific JSON fields.

| Field | Type | Meaning |
|---|---|---|
| `feedback_id` | string | Globally namespaced identifier, currently `yandex_maps:<review_id>` |
| `source` | enum | Origin of the observation, currently `yandex_maps` |
| `source_record_id` | string | Original platform record identifier for traceability |
| `text` | string | Original feedback text used by later text processing |
| `rating` | integer | Source rating in `1..5` |
| `published_on` | date | Source publication calendar date |
| `organization_id` | string | Stable source organization identifier |
| `organization_name` | string | Organization display name at collection time |
| `organization_url` | string | Provenance URL for the organization |
| `search_query` | string | Query that selected the source organization, when available |

The canonical analytical record intentionally has no author field. A display name is neither a stable customer ID
nor required for the agreed first-version metrics, so propagating it would add privacy exposure without analytical
value. This means record counts must never be presented as unique-customer counts.

Canonical datasets are accompanied by a strict `FeedbackPreparationManifest`. It binds the source and output using
SHA-256 checksums and records their sizes, creation time, observation count, organization count, rating distribution,
and publication-date coverage. Preparation refuses duplicate global feedback identifiers and existing destination
files. The output and manifest are written through temporary files so invalid or interrupted input is not published as
a completed artifact.

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
| `comment_replies` | array | Inline replies with required `text` and optional metadata |
| `video_id` | string | YouTube video identifier |
| `video_title` | string | Video title at export time |
| `video_channel` | string | Channel title at export time |
| `video_url` | string | Source video URL |
| `search_query` | string | Query that first discovered the source |

## Text units and cleaned development output

Before cleaning, each exported thread is flattened into unambiguous records. `text_kind` is either `comment` or
`reply`; replies require `parent_record_id`. `record_id` is the platform ID when available, otherwise a deterministic
hash. Author, timestamp, likes, and source provenance are optional and never determine text ownership.

`development-clean.jsonl` adds `clean_text`, `detected_language`, and `duplicate_count` while retaining the original
`text`. The original supports audit and provenance; only `clean_text` is intended for embeddings. Exact duplicates are
represented once, and `duplicate_count` records the normalized group size.

## Historical cluster import (`clusters.jsonl`)

One non-outlier topic per line. This compatibility format is accepted by `/analysis` for existing results but is no
longer produced by a supported repository command. New integrations consume the stage-13 snapshot.

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

## Historical run metadata (`run_meta.json`)

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
- The compatibility Pydantic contracts, fixtures, and `/analysis` parser are covered by reader tests and must change
  together.
- No maintained CLI may produce the historical format or import its removed monolithic configuration.
