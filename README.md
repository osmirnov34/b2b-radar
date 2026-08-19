# b2b-radar

Automated B2B startup idea discovery from public discussions (YouTube, Reddit, forums, sites)

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Git
- [uv](https://docs.astral.sh/uv/) (for local development)

### Installation

```bash
git clone https://github.com/osmirnov34/b2b-radar
cd b2b-radar
cp .env.example .env   # replace development passwords before production use
docker compose up -d   # postgres -> migrations -> web
```

## Development

### Web UI

FastAPI + server-rendered Jinja2 templates, backed by the same Postgres database as the CLI pipeline.

```bash
uvicorn src.web.app:app --reload --port 8000   # http://127.0.0.1:8000
```

Pages: dashboard, sources/videos (`/videos`), video detail, comments (`/comments`), API keys (`/api-keys`),
and analysis imports (`/analysis`). Adding a source or reprocessing triggers `src/ingestion/pipeline.py` in a FastAPI background
task. Each source tracks `ingest_status` (`pending`, `running`, `success`, or `failed`).

The pipeline fetches up to 5 replies per comment via YouTube API (`part=snippet,replies`); all replies
are shown collapsed under comments on detail and `/comments` pages and included in JSONL export
(`comment_replies`, `comment_total_reply_count`).

### Docker

```bash
docker compose build
docker compose up                         # postgres, migrations, and web UI on :8000
docker compose --profile cli run --rm app python app.py "crm для малого бизнеса"
docker compose --profile cli run --rm app bash   # interactive
```

### Database Migrations

Alembic, scripts in `migrations/versions/`.

```bash
uv run alembic upgrade head                                  # apply
uv run alembic downgrade -1                                  # rollback last
uv run alembic revision --autogenerate -m "message"          # review before committing
uv run alembic current
uv run alembic history
```

Related tables can share one migration (e.g. FK dependencies) as long as `upgrade`/`downgrade` order respects them. Otherwise, one migration per change — don't edit an already-applied one.

### Offline topic analysis

Install the optional ML stack and validate the versioned pipeline configuration against an exported comments JSONL:

```bash
uv sync --extra analysis
uv run python scripts/run_ml_pipeline.py dry-run configs/pipeline.example.json
```

The supported workflow is `dry-run` → `smoke-run` → `run` → manual evaluation → export → publish. Embedding models
are downloaded on first use. Input, historical-import, and current stage contracts are documented in
[`docs/analysis-contracts.md`](docs/analysis-contracts.md); typed stage configuration is described in
[`docs/analysis-configuration.md`](docs/analysis-configuration.md).
Instructions for validating, sampling, and splitting the full corpus without source-group leakage are in
[`docs/dataset-preparation.md`](docs/dataset-preparation.md).
The same guide covers privacy-safe development EDA, explicit comment/reply text units, cleaning, and the thin notebook
in `notebooks/01_development_eda.ipynb`.

Before a real ML run, execute the notebook's mandatory preflight section or run
`python3 scripts/run_ml_pipeline.py dry-run configs/pipeline.example.json`. A real `run` repeats the same read-only
checks automatically and refuses to start when any check is blocked; see `docs/ml-operations.md`.

For an executable wiring check before allocating a full run, use
`python3 scripts/run_ml_pipeline.py smoke-run configs/pipeline.example.json --records 2000`. Smoke artifacts are local,
explicitly non-publishable, and stop before manual evaluation and export.
ANN-based semantic deduplication is documented in
[`docs/semantic-deduplication.md`](docs/semantic-deduplication.md).
The complete resumable pipeline, manual evaluation gate, atomic ML snapshot publication, rollback, retention policy,
and CI smoke checks are documented in [`docs/ml-operations.md`](docs/ml-operations.md).

### Deployment (Caddy + HTTP Basic Auth)

The `caddy` service (under the `prod` [profile](https://docs.docker.com/compose/how-tos/profiles/), so it's skipped in local dev) reverse-proxies the `web` UI behind HTTP Basic Auth and terminates HTTPS via Let's Encrypt. Requires the domain's DNS pointed at the host with ports 80/443 open. Config lives in `Caddyfile`.

```bash
# generate a bcrypt hash for BASIC_AUTH_HASH (keep the plaintext out of .env)
docker run --rm caddy:2-alpine caddy hash-password --plaintext 'your-password'

# set DOMAIN, BASIC_AUTH_USER, BASIC_AUTH_HASH in .env, then:
docker compose --profile prod up -d
```

The production endpoint is `https://$DOMAIN:8443`. Set `CLOUDFLARE_API_TOKEN` for the DNS challenge. To rotate the
password, regenerate the hash, update `BASIC_AUTH_HASH`, and run `docker compose restart caddy`.

### Code Quality

Install the project and developer tools, then enable pre-commit:

```bash
uv sync --extra dev
uv run pre-commit install
```

Pre-commit runs Ruff + mypy (strict) on `git commit`. The same checks can be run manually:

```bash
uv run ruff check .
uv run mypy src app.py
uv run pytest
uv run pre-commit run --all-files
git commit --no-verify   # bypass, not recommended
```

Unit tests do not need external services. Database integration tests use a disposable, dedicated PostgreSQL instance:

```bash
docker compose --profile test up -d --wait postgres-test
TEST_DATABASE_URL=postgresql+asyncpg://b2b_radar_test:b2b_radar_test@localhost:5434/b2b_radar_test \
  uv run pytest -m integration
docker compose --profile test stop postgres-test
```
