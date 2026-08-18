import json
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.ml.export import ExportConfig, export_topic_results
from src.web.ml_snapshot import MLSnapshotRepository, SnapshotUnavailableError, load_public_snapshot
from src.web.routers.ml_topics import router
from tests.test_export import _inputs


def _public_export(tmp_path: Path) -> Path:
    inputs = _inputs(tmp_path)
    output = tmp_path / "public-export"
    export_topic_results(*inputs, output)
    return output / "export-manifest.json"


def test_snapshot_requires_explicit_unreliable_policy_and_loads_public_contract(tmp_path: Path) -> None:
    manifest = _public_export(tmp_path)
    with pytest.raises(ValueError, match="preliminary"):
        load_public_snapshot(manifest)

    snapshot = load_public_snapshot(manifest, allow_unreliable=True)
    assert len(snapshot.topics) == 2
    assert len(snapshot.assignments) == 7
    assert len(snapshot.assignments_by_topic[0]) == 3
    assert snapshot.assignments_by_topic[None] == ()


def test_snapshot_rejects_research_manifest_and_paths_outside_export_directory(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    research = tmp_path / "research"
    export_topic_results(*inputs, research, config=ExportConfig(include_research_text=True))
    with pytest.raises(ValueError, match="sensitive research"):
        load_public_snapshot(research / "export-manifest.json", allow_unreliable=True)

    manifest = _public_export(tmp_path / "confined")
    data = json.loads(manifest.read_text())
    data["topics_path"] = str(tmp_path / "outside" / "topics.jsonl")
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="beside export-manifest"):
        load_public_snapshot(manifest, allow_unreliable=True)


def test_repository_keeps_last_known_good_snapshot_after_failed_reload(tmp_path: Path) -> None:
    manifest = _public_export(tmp_path)
    repository = MLSnapshotRepository(manifest, enabled=True, allow_unreliable=True)
    original = repository.get_snapshot()
    topics = manifest.parent / "topics.jsonl"
    topics.write_text(topics.read_text() + "\n")
    current_mtime = manifest.stat().st_mtime_ns
    os.utime(manifest, ns=(current_mtime + 1, current_mtime + 1))

    assert repository.get_snapshot() is original
    assert repository.last_error == "ML snapshot reload failed public-contract validation"


def test_disabled_or_missing_repository_is_nonfatal(tmp_path: Path) -> None:
    with pytest.raises(SnapshotUnavailableError, match="disabled"):
        MLSnapshotRepository(None, enabled=False).get_snapshot()
    with pytest.raises(SnapshotUnavailableError, match="not configured"):
        MLSnapshotRepository(None, enabled=True).get_snapshot()
    with pytest.raises(SnapshotUnavailableError):
        MLSnapshotRepository(tmp_path / "missing.json", enabled=True).get_snapshot()


async def test_ml_api_lists_topics_assignments_quality_and_handles_missing_topic(tmp_path: Path) -> None:
    repository = MLSnapshotRepository(_public_export(tmp_path), enabled=True, allow_unreliable=True)
    app = FastAPI()
    app.state.ml_repository = repository
    app.include_router(router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        topics = await client.get("/api/topics")
        assert topics.status_code == 200
        assert topics.json()["total"] == 2
        assert (await client.get("/api/topics/0")).status_code == 200
        assignments = await client.get("/api/topics/0/assignments?limit=2")
        assert assignments.status_code == 200
        assert len(assignments.json()["items"]) == 2
        assert (await client.get("/api/topics/999")).status_code == 404
        quality = await client.get("/api/ml/quality")
        assert quality.status_code == 200
        assert quality.json()["quality"]["preliminary"] is True
        assert (await client.get("/api/ml/outliers")).json()["total"] == 0
