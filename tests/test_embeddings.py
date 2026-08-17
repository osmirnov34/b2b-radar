import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.ml.config import DeduplicationConfig, EmbeddingConfig
from src.ml.embeddings import embedding_prompt, generate_embeddings
from src.ml.schemas import EmbeddingArtifactManifest
from src.ml.semantic_deduplication import run_semantic_deduplication


class FakeEncoder:
    device = "cpu"

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        if self.calls == self.fail_on_call:
            message = "simulated interruption"
            raise RuntimeError(message)
        rows = []
        for text in texts:
            raw = np.asarray([len(text), sum(map(ord, text)) % 101 + 1, 1], dtype=np.float32)
            rows.append(raw / np.linalg.norm(raw))
        return np.asarray(rows, dtype=np.float32)


def _write_cleaned(path: Path, count: int = 5, *, duplicate_id: bool = False) -> None:
    rows = [
        {
            "record_id": "record-0" if duplicate_id else f"record-{index}",
            "text": f"Raw text {index}",
            "clean_text": f"Clean business problem {index}",
            "text_kind": "comment",
            "detected_language": "ru",
        }
        for index in range(count)
    ]
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")


def test_generate_embeddings_writes_aligned_artifacts(tmp_path: Path) -> None:
    records = tmp_path / "clean.jsonl"
    _write_cleaned(records)
    output = tmp_path / "embeddings"
    config = EmbeddingConfig(model_name="fake/e5", batch_size=2)

    manifest = generate_embeddings(records, output, config=config, encoder=FakeEncoder())

    vectors = np.load(output / "embeddings.npy", allow_pickle=False)
    ids = [json.loads(line) for line in (output / "record-ids.jsonl").read_text().splitlines()]
    persisted = EmbeddingArtifactManifest.model_validate_json((output / "embedding-manifest.json").read_text())
    assert vectors.shape == (5, 3)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert ids == [f"record-{index}" for index in range(5)]
    assert persisted == manifest
    assert manifest.records_sha256 == hashlib.sha256(records.read_bytes()).hexdigest()
    assert manifest.record_ids_sha256 == hashlib.sha256((output / "record-ids.jsonl").read_bytes()).hexdigest()
    assert not list(output.glob(".*partial*"))


def test_generate_embeddings_resumes_at_completed_batch(tmp_path: Path) -> None:
    records = tmp_path / "clean.jsonl"
    _write_cleaned(records)
    output = tmp_path / "embeddings"
    config = EmbeddingConfig(model_name="fake/model", batch_size=2)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        generate_embeddings(records, output, config=config, encoder=FakeEncoder(fail_on_call=2))

    checkpoint = json.loads((output / ".embedding-progress.json").read_text())
    assert checkpoint["processed_records"] == 2
    resumed_encoder = FakeEncoder()
    manifest = generate_embeddings(records, output, config=config, encoder=resumed_encoder, resume=True)
    assert manifest.n_records == 5
    assert resumed_encoder.calls == 2
    assert not (output / ".embedding-progress.json").exists()


def test_resume_rejects_changed_configuration(tmp_path: Path) -> None:
    records = tmp_path / "clean.jsonl"
    _write_cleaned(records)
    output = tmp_path / "embeddings"
    config = EmbeddingConfig(model_name="fake/model", batch_size=2)
    with pytest.raises(RuntimeError):
        generate_embeddings(records, output, config=config, encoder=FakeEncoder(fail_on_call=2))

    changed = EmbeddingConfig(model_name="another/model", batch_size=2)
    with pytest.raises(ValueError, match="changed input data or embedding configuration"):
        generate_embeddings(records, output, config=changed, encoder=FakeEncoder(), resume=True)


def test_generate_embeddings_rejects_duplicate_ids_and_bad_vectors(tmp_path: Path) -> None:
    records = tmp_path / "duplicates.jsonl"
    _write_cleaned(records, count=2, duplicate_id=True)
    with pytest.raises(ValueError, match="duplicate record_id"):
        generate_embeddings(records, tmp_path / "duplicate-output", encoder=FakeEncoder())

    _write_cleaned(records, count=2)

    class BadEncoder(FakeEncoder):
        def encode(self, texts: list[str]) -> np.ndarray:
            return np.full((len(texts), 2), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="NaN or infinite"):
        generate_embeddings(records, tmp_path / "bad-output", encoder=BadEncoder())


def test_embedding_manifest_is_accepted_by_semantic_deduplication(tmp_path: Path) -> None:
    records = tmp_path / "clean.jsonl"
    _write_cleaned(records, count=3)
    embedding_dir = tmp_path / "embeddings"
    generate_embeddings(records, embedding_dir, encoder=FakeEncoder())

    result = run_semantic_deduplication(
        records,
        embedding_dir / "embeddings.npy",
        embedding_dir / "embedding-manifest.json",
        tmp_path / "deduplication",
        config=DeduplicationConfig(backend="exhaustive", threshold=1.0),
    )
    assert result.result.n_input == 3


def test_embedding_prompt_uses_explicit_value_and_model_defaults() -> None:
    assert embedding_prompt(EmbeddingConfig(model_name="intfloat/multilingual-e5-large")) == "query: "
    assert embedding_prompt(EmbeddingConfig(model_name="ai-forever/FRIDA")) == "categorize_topic: "
    assert embedding_prompt(EmbeddingConfig(model_name="custom/model")) == ""
    assert embedding_prompt(EmbeddingConfig(model_name="custom/model", prompt_prefix="passage: ")) == "passage: "
