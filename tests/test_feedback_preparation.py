import hashlib
import json
from pathlib import Path

import pytest

from src.analytics import FeedbackPreparationManifest
from src.operations.feedback_preparation import FeedbackPreparationError, prepare_yandex_feedback


def _row(review_id: str, *, rating: int = 4, org_id: str = "229248496819") -> dict[str, object]:
    return {
        "review_id": review_id,
        "review_text": " Исходный отзыв без изменения ",
        "rating": rating,
        "author": "Не должен попасть в результат",
        "published_at": "2025-02-06",
        "org_id": org_id,
        "org_name": "Стоматология",
        "org_url": f"https://yandex.ru/maps/org/{org_id}/reviews/",
        "search_query": "стоматология",
    }


def _write_source(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_prepare_yandex_feedback_writes_canonical_jsonl_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "interim" / "feedback.jsonl"
    manifest_path = tmp_path / "interim" / "feedback.manifest.json"
    _write_source(
        source,
        [
            _row("7995cc14c8b92196", rating=2),
            _row("6529392e51d48c89", rating=4, org_id="987654321"),
        ],
    )

    manifest = prepare_yandex_feedback(source, output, manifest_path)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    persisted_manifest = FeedbackPreparationManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    assert manifest == persisted_manifest
    assert manifest.records == 2
    assert manifest.unique_organizations == 2
    assert manifest.rating_distribution == {2: 1, 4: 1}
    assert manifest.output_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert records[0]["feedback_id"] == "yandex_maps:7995cc14c8b92196"
    assert records[0]["text"] == " Исходный отзыв без изменения "
    assert "author" not in records[0]


def test_prepare_yandex_feedback_refuses_existing_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "feedback.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_source(source, [_row("7995cc14c8b92196")])
    output.write_text("keep", encoding="utf-8")

    with pytest.raises(FeedbackPreparationError, match="refusing to replace"):
        prepare_yandex_feedback(source, output, manifest)

    assert output.read_text(encoding="utf-8") == "keep"
    assert not manifest.exists()


def test_prepare_yandex_feedback_removes_partial_output_on_duplicate_id(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "feedback.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_source(source, [_row("7995cc14c8b92196"), _row("7995cc14c8b92196")])

    with pytest.raises(FeedbackPreparationError, match="duplicate feedback identifier") as captured:
        prepare_yandex_feedback(source, output, manifest)

    assert "7995cc14c8b92196" not in str(captured.value)
    assert not output.exists()
    assert not manifest.exists()
    assert not list(tmp_path.glob(".*.tmp"))
