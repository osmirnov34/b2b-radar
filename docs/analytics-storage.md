# Analytical PostgreSQL storage

The reporting extension uses the dedicated PostgreSQL schema `analytics`. Its SQLAlchemy metadata is separate from
the operational application tables, so analytical evolution does not couple ingestion or the historical cluster
viewer to the reporting product. Alembic still applies both metadata collections through the repository's single
migration chain.

## Snapshot model

```text
analytics.datasets
        │ 1
        ├────────── * analytics.organizations
        │                         │ 1
        └─────────────────────────┴──── * analytics.feedback
                 │ 1                         │ 1
                 └──── * analytics.problem_runs ──── * analytics.problem_assessments
                                      │ 1
                                      └──── * analytics.problem_categories
```

- `datasets` identifies an immutable canonical artifact by SHA-256 and preserves its preparation metadata.
- `organizations` stores the organization dimension inside a dataset snapshot. The same source organization may have
  a different display name in a later snapshot without rewriting history.
- `feedback` is the private fact table. Its primary key is `(dataset_id, feedback_id)`, so repeated imports from
  different dataset versions remain distinguishable. A second uniqueness constraint prevents duplicate source records
  inside one snapshot.
- `problem_runs` registers one immutable, versioned assessment artifact against an exact feedback snapshot.
- `problem_assessments` stores category, stance, confidence, method, review state, and evidence offsets. It references
  the private feedback row but never copies review text.
- `problem_categories` freezes Russian labels, definitions, and business-impact hypotheses for the exact taxonomy
  version used by the run. Historical BI output therefore cannot silently adopt a newer taxonomy.

Deleting a dataset cascades only through its own organizations and feedback. Rating and aggregate consistency checks
are enforced by PostgreSQL. Indexes support the initial DataLens filters by dataset, organization, date, and rating.

## Privacy boundary

`analytics.feedback.text` contains private source text and must not be exposed directly through a public DataLens
connection. Author display names are absent from all analytical tables. Later public marts must use aggregates and
explicitly reviewed examples. A feedback row is an observation, not proof of a unique customer.

## Local migration

With the repository's local PostgreSQL configuration running:

```bash
docker compose up -d --wait postgres
docker compose run --rm migrate
```

This stage defines and migrates storage only. Loading a prepared JSONL and reconciling its manifest are separate
operations so an invalid or partial import cannot be mistaken for a registered dataset.

## Loading a snapshot

Apply migrations, then load the private canonical artifact together with its manifest:

```bash
uv run python scripts/load_analytics_feedback.py \
  data/interim/yandex-feedback.jsonl \
  data/interim/yandex-feedback.jsonl.manifest.json
```

Before opening a database transaction, the loader verifies the artifact size and SHA-256, parses every record, checks
record and organization counts, rating distribution, publication range, source identity, unique identifiers, and
consistent organization metadata. It then inserts the dataset, organizations, and feedback in one transaction using
batches of 1,000 rows. A failure rolls back the complete snapshot. Repeating the same verified artifact is idempotent:
the existing dataset ID is returned only after its stored row counts have been reconciled with the manifest.

## Reconciling stored content

After loading, compare every canonical record with PostgreSQL and optionally save an aggregate JSON report:

```bash
uv run python scripts/reconcile_analytics_feedback.py \
  data/interim/yandex-feedback.jsonl \
  data/interim/yandex-feedback.jsonl.manifest.json \
  --report data/reports/yandex-feedback-reconciliation.json
```

Reconciliation verifies dataset metadata, organization and feedback counts, ratings, date coverage, and a SHA-256
fingerprint of every reconstructed canonical record. It reports only counts of missing, unexpected, or changed rows;
record identifiers and feedback text are never written to the report. A non-matching or missing snapshot returns a
non-zero process status. Reports are created with exclusive mode and cannot overwrite an earlier validation result.

## Loading problem assessments

After loading the matching canonical feedback snapshot, import the checksum-verified output from stage 10:

```bash
uv run python scripts/load_problem_assessments.py \
  data/analytics/yandex-feedback-problems.jsonl \
  data/analytics/yandex-feedback-problems.jsonl.manifest.json
```

The preflight reconciles taxonomy, method, stance, category, and unique-key counts. Inside one transaction the loader
resolves the exact dataset by feedback SHA-256, verifies every feedback foreign key and every evidence offset against
the private source text length, then inserts the run, taxonomy snapshot, and assessments. Repeating the same artifact
is idempotent. If the taxonomy was moved, pass its verified location with `--taxonomy`.

## Aggregate KPI views

Stage 12 exposes aggregate-only PostgreSQL views suitable for the first DataLens dashboard:

- `analytics.organization_kpis`: review volume, average rating, rating distribution, low/high rating shares, and date
  coverage by organization;
- `analytics.problem_kpis`: category prevalence by organization and run, with `complaint`, `mixed`, and `praise`
  reported separately;
- `analytics.organization_monthly_kpis`: monthly review and rating dynamics;
- `analytics.problem_monthly_kpis`: monthly category dynamics with complaint prevalence.

Problem shares use all organization reviews in the same dataset (or month) as the denominator. A multi-label review
may occur in several category rows, so category shares must not be summed. `complaint_share` is the primary problem
indicator; `mixed_share` remains separate and `praise_share` describes strengths. Every view retains small groups but
sets `sample_sufficient = false` below 30 source reviews. Views contain no review text, evidence excerpts, or authors.
