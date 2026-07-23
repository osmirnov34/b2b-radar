# b2b-radar

Automated B2B startup idea discovery from public discussions (YouTube, Reddit, forums, sites)

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Git

### Installation

```bash
git clone https://github.com/osmirnov34/b2b-radar
cd b2b-radar
pre-commit install
cp .env.example .env   # fill in POSTGRES_* (used by Docker Compose and Alembic)
docker-compose up -d postgres
alembic upgrade head
```

## Development

### Web UI

FastAPI + server-rendered Jinja2 templates, backed by the same Postgres database as the CLI pipeline.

```bash
uvicorn src.web.app:app --reload --port 8000   # http://127.0.0.1:8000
```

Pages: dashboard, sources/videos (`/videos`), video detail, comments (`/comments`), API keys (`/api-keys`).
Adding a source or reprocessing a video triggers `src/pipeline.py` as a background task, so at least one
active row must exist in `youtube_api_keys` (add one from the `/api-keys` page).

### Docker

```bash
docker-compose build
docker-compose up          # starts postgres, the CLI `app` container, and the `web` UI on :8000
docker-compose run --rm app bash   # interactive
```

### Database Migrations

Alembic, scripts in `migrations/versions/`.

```bash
alembic upgrade head                                  # apply
alembic downgrade -1                                   # rollback last
alembic revision --autogenerate -m "message"           # after model changes — review the diff before committing
alembic current / alembic history                      # inspect
```

Related tables can share one migration (e.g. FK dependencies) as long as `upgrade`/`downgrade` order respects them. Otherwise, one migration per change — don't edit an already-applied one.

### Code Quality

pre-commit runs Ruff + mypy (strict) on `git commit`.

```bash
pre-commit run --all-files
pre-commit run mypy --all-files
pre-commit run ruff --all-files
pre-commit autoupdate
git commit --no-verify   # bypass, not recommended
```
