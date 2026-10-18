from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import SQLAlchemyError

from src.config import AnalyticsAPISettings
from src.web import analytics_app

# Test-only credentials; these are never used outside the in-process test app.
# ruff: noqa: S106


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    settings = AnalyticsAPISettings(analytics_api_token="aggregate-test", analytics_private_token="private-test")
    monkeypatch.setattr(analytics_app, "get_analytics_api_settings", lambda: settings)
    async with AsyncClient(transport=ASGITransport(app=analytics_app.app), base_url="http://test") as instance:
        yield instance
    analytics_app.app.dependency_overrides.clear()


async def test_access_checks_precede_database_access(client: AsyncClient) -> None:
    assert (await client.get("/api/analytics/summary?dataset_id=1")).status_code == 401
    response = await client.get(
        "/api/analytics/summary?dataset_id=1",
        headers={"Authorization": "Bearer invalid"},
    )
    assert response.status_code == 401
    response = await client.get(
        "/api/analytics/private/examples?run_id=1&category_id=waiting_time",
        headers={"Authorization": "Bearer aggregate-test"},
    )
    assert response.status_code == 401


async def test_unconfigured_access_is_closed(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AnalyticsAPISettings(analytics_api_token=None, analytics_private_token=None)
    monkeypatch.setattr(analytics_app, "get_analytics_api_settings", lambda: settings)
    assert (await client.get("/api/analytics/summary?dataset_id=1")).status_code == 503


async def test_equal_tokens_do_not_unlock_private_examples(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = AnalyticsAPISettings(analytics_api_token="same", analytics_private_token="same")
    monkeypatch.setattr(analytics_app, "get_analytics_api_settings", lambda: settings)
    response = await client.get(
        "/api/analytics/private/examples?run_id=1&category_id=waiting_time",
        headers={"Authorization": "Bearer same"},
    )
    assert response.status_code == 503


async def test_summary_response_whitelists_fields_and_missing_is_404(client: AsyncClient) -> None:
    session = AsyncMock()
    result = MagicMock()
    result.mappings.return_value.one_or_none.return_value = {
        "dataset_id": 1,
        "source": "yandex_maps",
        "feedback_count": 10,
        "organization_count": 2,
        "published_on_min": None,
        "published_on_max": None,
        "text": "PRIVATE MEDICAL TEXT",
        "artifact_sha256": "private",
    }
    session.execute.return_value = result

    async def override() -> AsyncMock:
        return session

    analytics_app.app.dependency_overrides[analytics_app.get_analytics_session] = override
    response = await client.get(
        "/api/analytics/summary?dataset_id=1", headers={"Authorization": "Bearer aggregate-test"}
    )
    assert response.status_code == 200
    assert response.json()["feedback_count"] == 10
    assert "PRIVATE" not in response.text
    assert "artifact_sha256" not in response.text
    result.mappings.return_value.one_or_none.return_value = None
    response = await client.get(
        "/api/analytics/summary?dataset_id=99", headers={"Authorization": "Bearer aggregate-test"}
    )
    assert response.status_code == 404


async def test_api_has_only_read_routes_and_bounded_pagination(client: AsyncClient) -> None:
    schema = (await client.get("/openapi.json")).json()
    assert all(set(methods) == {"get"} for methods in schema["paths"].values())
    session = AsyncMock()

    async def override() -> AsyncMock:
        return session

    analytics_app.app.dependency_overrides[analytics_app.get_analytics_session] = override
    for query in ("dataset_id=0", "dataset_id=1&limit=201", "dataset_id=1&offset=-1"):
        response = await client.get(
            "/api/analytics/organizations?" + query,
            headers={"Authorization": "Bearer aggregate-test"},
        )
        assert response.status_code == 422
    session.execute.assert_not_called()
    assert (await client.post("/api/analytics/summary")).status_code == 405


async def test_database_error_is_sanitized(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def failed_session() -> AsyncIterator[AsyncMock]:
        session = AsyncMock()
        session.execute.side_effect = SQLAlchemyError("SQL containing PRIVATE TEXT")
        session.begin = MagicMock(return_value=session)
        yield session

    monkeypatch.setattr(analytics_app, "get_session", failed_session)
    response = await client.get(
        "/api/analytics/summary?dataset_id=1", headers={"Authorization": "Bearer aggregate-test"}
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "Analytical storage is unavailable"}
    assert "PRIVATE TEXT" not in response.text
