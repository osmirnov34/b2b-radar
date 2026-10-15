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
```

- `datasets` identifies an immutable canonical artifact by SHA-256 and preserves its preparation metadata.
- `organizations` stores the organization dimension inside a dataset snapshot. The same source organization may have
  a different display name in a later snapshot without rewriting history.
- `feedback` is the private fact table. Its primary key is `(dataset_id, feedback_id)`, so repeated imports from
  different dataset versions remain distinguishable. A second uniqueness constraint prevents duplicate source records
  inside one snapshot.

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
