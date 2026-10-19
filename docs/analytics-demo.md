# Локальная демонстрация и приёмка

Команды выполняются из корня проекта. Нужны Docker, зависимости проекта с `dev` и `reporting`,
и шрифт `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`. Это отдельная локальная среда, не облачный релиз.

## Первый запуск

```bash
.venv/bin/python scripts/manage_analytics_demo.py init --source /полный/путь/reviews.jsonl
docker compose -p b2b-radar-analytics-demo --env-file data/demo/analytics-local/environment.env -f compose.analytics-demo.yml up -d --wait
.venv/bin/python scripts/manage_analytics_demo.py migrate
.venv/bin/python scripts/manage_analytics_demo.py check
.venv/bin/python scripts/manage_analytics_demo.py run
.venv/bin/python scripts/manage_analytics_demo.py bi-role
.venv/bin/python scripts/manage_analytics_demo.py resume
.venv/bin/python scripts/manage_analytics_demo.py accept
```

`init` создаёт новую папку с правами 0700 и случайными секретами в `environment.env` (0600).
Повторный `init` запрещён. БД `analytics_demo` доступна только на `127.0.0.1:55440`,
её данные сохраняются в отдельном Docker volume. Скрипт отклоняет конфигурацию с нелокальным адресом БД.
`migrate` применяет настоящую цепочку Alembic; перед изменением уже используемой базы нужна резервная копия.

`bi-role` создаёт NOLOGIN-роль `b2b_radar_bi_reader` для проверки прав; команду выполняют один раз.
Это не логин для подключения DataLens. В облаке потребуется отдельная LOGIN-учётная запись с этой ролью.
Не выдавайте BI-пользователю права на исходные таблицы `analytics`.

Если выполнение прервалось, используйте `resume`, а не повторный `run`: завершённые файлы проверяются,
а не перезаписываются. Для нового исследования задайте новый `run_name` в `pipeline.json`.
После изменения подписанного кода процесс потребует новый запуск; не редактируйте подписи вручную.

## Где результат

В `data/demo/analytics-local/` (папка игнорируется Git):

- `environment.env` — пароль БД и два различных токена API; не публиковать;
- `pipeline.json` — конфигурация источника и имени запуска;
- `acceptance.json` — агрегатный протокол проверки, создаётся без перезаписи;
- `runs/<run_name>/pipeline-state.json` — состояние и ID в базе;
- `runs/<run_name>/01-prepare/` — приватные канонические отзывы;
- `runs/<run_name>/07-datalens/package/` — SQL-источники и спецификация дашборда;
- `runs/<run_name>/08-pdf/report/report.pdf` — управленческий отчёт;
- там же `report.json` и `manifest.json` — агрегаты и контрольные суммы отчёта.

`accept` сверяет файлы с контрольными точками, версию миграций, числа в БД/PDF,
доступ к витринам и запрет доступа BI-роли к текстам. Он также проверяет шесть агрегатных API-маршрутов,
отказ анонимному клиенту и отказ агрегатному токену в доступе к приватным примерам.
Проверка API выполняется внутри процесса без открытого сетевого порта.

## Просмотр API

```bash
.venv/bin/python scripts/manage_analytics_demo.py serve
```

Откройте `http://127.0.0.1:8001/docs`. Для агрегатов используйте Authorize с `ANALYTICS_API_TOKEN`
из локального файла, затем, например, `/api/analytics/summary` с `dataset_id` из `pipeline-state.json`.
Приватные примеры требуют другого токена и доступны только для принятых оценок.
Сервер привязан к loopback; не публикуйте его административные учётные данные.

## Остановка и продолжение

```bash
docker compose -p b2b-radar-analytics-demo --env-file data/demo/analytics-local/environment.env -f compose.analytics-demo.yml stop
docker compose -p b2b-radar-analytics-demo --env-file data/demo/analytics-local/environment.env -f compose.analytics-demo.yml up -d --wait
```

Остановка не удаляет volume. Не используйте `down -v`, если нужны сохранённые данные.
Локальную БД нельзя напрямую подключить к облачному DataLens. Облачная PostgreSQL, TLS,
сетевые ограничения, отдельный BI-логин и ручная сборка дашборда — отдельные шаги
по [инструкции DataLens](datalens-dashboard.md).

Режим по умолчанию — `rules_only`: проверка техники не подтверждает качество разметки,
не заменяет ручную оценку, не измеряет финансовый ущерб и не означает готовность публичного продукта.

## Результат локальной приёмки 19.10.2026

Контрольный запуск `acceptance-003` завершён: 21 630 отзывов, 720 организаций, 10 категорий,
`dataset_id=1`, `problem_run_id=2`. Завершённый resume не изменил артефакты.
Протокол `acceptance.json` подтвердил миграции, контрольные точки, API и изоляцию BI-доступа.
PDF сформирован, первая страница визуально проверена.

До обновления статистики запрос категорий был отменён после более трёх минут.
Точный запрос категорий PDF после исправления выполнен трижды за 1612, 1527 и 1592 мс
на этой локальной БД. Это наблюдение на текущем датасете, не гарантия SLA для других объёмов.
Числа отзывов, организаций, категорий, помесячные агрегаты и рекомендации полностью совпали
с предыдущим успешно сформированным отчётом.

Проверки: 374 теста прошли, 1 пропущен из-за отсутствия optional-зависимости SciPy;
Ruff и строгий mypy прошли. Облачное развёртывание и сборка дашборда в аккаунте DataLens
остаются отдельной незавершённой частью приёмки.
