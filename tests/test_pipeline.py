from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.domain.source import IngestStatus, Source, SourceType
from src.ingestion.pipeline import _save_documents


def _source(source_id: int) -> Source:
    return Source(
        id=source_id,
        type=SourceType.YOUTUBE_VIDEO,
        url=f"https://www.youtube.com/watch?v={source_id}",
        name="Video",
        extracted_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_save_documents_marks_source_success(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = SimpleNamespace(
        session=SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock()),
        source_repo=SimpleNamespace(set_ingest_status=AsyncMock()),
    )
    monkeypatch.setattr("src.ingestion.pipeline._extract_documents_for_source", AsyncMock(return_value=4))

    saved = await _save_documents(ctx, [_source(1)], 100)  # type: ignore[arg-type]

    assert saved == 4
    assert ctx.source_repo.set_ingest_status.await_args_list[0].args == (1, IngestStatus.RUNNING)
    assert ctx.source_repo.set_ingest_status.await_args_list[1].args == (1, IngestStatus.SUCCESS)
    assert ctx.session.rollback.await_count == 0
    assert ctx.session.commit.await_count == 2


@pytest.mark.asyncio
async def test_save_documents_isolates_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = SimpleNamespace(
        session=SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock()),
        source_repo=SimpleNamespace(set_ingest_status=AsyncMock()),
    )
    monkeypatch.setattr(
        "src.ingestion.pipeline._extract_documents_for_source",
        AsyncMock(side_effect=RuntimeError("quota")),
    )

    saved = await _save_documents(ctx, [_source(1), _source(2)], 100)  # type: ignore[arg-type]

    assert saved == 0
    assert ctx.session.rollback.await_count == 2
    failed_calls = ctx.source_repo.set_ingest_status.await_args_list[1::2]
    assert all(
        call.args == (source_id, IngestStatus.FAILED, "quota")
        for source_id, call in zip((1, 2), failed_calls, strict=True)
    )
