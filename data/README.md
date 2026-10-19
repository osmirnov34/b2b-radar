# Local analysis data

This directory is intentionally excluded from Git except for this guide.

- `raw/` — immutable downloaded source files.
- `interim/` — reproducible development/validation/test splits and other intermediate files.
- `processed/` — cleaned data, embeddings, deduplication, corpus, UMAP, clusters, topics, and final outlier decisions.
- `samples/` — deterministic research subsets and metadata.
- `reports/` — dataset profiles and sanitized validation errors.

Run a privacy-safe structural audit of a downloaded Yandex Maps reviews export before adapting or processing it:

```bash
uv run python scripts/audit_yandex_reviews.py data/raw/yandex-reviews.jsonl \
  --report-dir data/reports/yandex-reviews \
  --minimum-reviews 30
```

The JSON and Markdown reports contain aggregate field, identifier, organization, rating, date, duplicate, and text
length statistics. They intentionally exclude review text and author values. The source file is read-only and is never
rewritten by the audit.

After a successful audit, create a source-neutral private analytical artifact at a new path:

```bash
uv run python scripts/prepare_yandex_feedback.py \
  data/raw/yandex-reviews.jsonl \
  data/interim/yandex-feedback.jsonl
```

The command also writes `data/interim/yandex-feedback.jsonl.manifest.json` with input/output checksums, record and
organization counts, rating distribution, and date coverage. It never replaces existing files: use a new output path
for a new dataset version or run. The canonical JSONL retains review text for downstream analysis but excludes author
display names. Treat both the JSONL and its manifest as private local artifacts and do not commit them.

After applying the PostgreSQL migrations, load the prepared pair transactionally:

```bash
uv run python scripts/load_analytics_feedback.py \
  data/interim/yandex-feedback.jsonl \
  data/interim/yandex-feedback.jsonl.manifest.json
```

The loader validates the entire artifact before writing, imports one snapshot in a single transaction, and safely
recognizes an already loaded checksum. It never logs review text or author values.

Verify the persisted snapshot after loading:

```bash
uv run python scripts/reconcile_analytics_feedback.py \
  data/interim/yandex-feedback.jsonl \
  data/interim/yandex-feedback.jsonl.manifest.json \
  --report data/reports/yandex-feedback-reconciliation.json
```

The report contains aggregate differences only and is safe from raw feedback text. Keep it with the matching dataset
version so later metrics and dashboards can prove which PostgreSQL snapshot they used.

Never commit comments, embeddings, generated reports, or model artifacts.
Topic keywords and generated names can contain sensitive terms derived from comments; review them before export.

# Приоритеты проблем

После применения миграций и загрузки оценок можно получить рекомендации для выбранного запуска:

```bash
.venv/bin/python scripts/rank_analytics_problems.py 1
```

`1` — ID в `analytics.problem_runs`. Формула, статусы проверки и ограничения описаны в
[документации приоритетов](../docs/problem-priorities.md).

## Пакет для DataLens

После подготовки запуска разметки создайте SQL-источники и спецификацию графиков:

```bash
.venv/bin/python scripts/prepare_datalens_dashboard.py 1 data/reports/datalens-run-1
```

Здесь `1` — ID из `analytics.problem_runs`. Пакет не содержит данных или паролей и не создаёт графики в аккаунте.
Порядок подключения и ручной сборки описан в [инструкции DataLens](../docs/datalens-dashboard.md).

## Управленческий PDF

```bash
.venv/bin/python scripts/generate_analytical_report.py 1 data/reports/management-run-1
```

Требуются reporting-extra и TTF-шрифт с кириллицей. PDF, агрегатный JSON и манифест создаются в новой папке.
Подробности: [генерация отчёта](../docs/analytical-pdf-report.md).

## Единый запуск

```bash
.venv/bin/python scripts/run_analytics_pipeline.py configs/analytics-pipeline.example.json --dry-run
.venv/bin/python scripts/run_analytics_pipeline.py configs/analytics-pipeline.example.json
```

Предварительно задайте путь к данным и уникальный run_name в конфигурации. Стадии и продолжение после остановки
описаны в [инструкции сквозного запуска](../docs/analytics-pipeline.md).

Для отдельной локальной PostgreSQL, реального PDF и проверки доступа без облачных ресурсов используйте
[инструкцию демо-приёмки](../docs/analytics-demo.md).
