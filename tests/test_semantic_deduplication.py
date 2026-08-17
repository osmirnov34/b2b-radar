import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.semantic_deduplicate import main
from src.ml.config import DeduplicationConfig
from src.ml.schemas import CleanedTextUnit, EmbeddingArtifactManifest, TextKind
from src.ml.semantic_deduplication import (
    ExhaustiveCandidateIndex,
    SemanticDeduplicationManifest,
    run_semantic_deduplication,
    semantic_deduplicate,
)


def _config(**overrides: object) -> DeduplicationConfig:
    return DeduplicationConfig.model_validate(
        {
            "threshold": 0.95,
            "backend": "exhaustive",
            "ann_neighbors": 8,
            **overrides,
        },
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_semantic_dedup_keeps_roles_separate_and_earliest_representative() -> None:
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.999, 0.01],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    roles = [TextKind.COMMENT, TextKind.COMMENT, TextKind.REPLY, TextKind.COMMENT]

    result = semantic_deduplicate(
        vectors,
        _config(),
        text_kinds=roles,
        candidate_index=ExhaustiveCandidateIndex(),
    )

    assert result.keep_indices == [0, 2, 3]
    assert len(result.groups) == 1
    assert result.groups[0].representative_index == 0
    assert result.groups[0].duplicate_indices == [1]
    assert result.stats.n_removed == 1


def test_semantic_dedup_uses_transitive_components() -> None:
    angles = np.deg2rad([0, 20, 40])
    vectors = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)

    result = semantic_deduplicate(
        vectors,
        _config(threshold=0.90),
        candidate_index=ExhaustiveCandidateIndex(),
    )

    assert result.keep_indices == [0]
    assert result.groups[0].duplicate_indices == [1, 2]


def test_hnsw_backend_finds_obvious_duplicates_and_respects_roles() -> None:
    pytest.importorskip("hnswlib")
    vectors = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    roles = [TextKind.COMMENT, TextKind.COMMENT, TextKind.REPLY, TextKind.COMMENT]

    result = semantic_deduplicate(vectors, DeduplicationConfig(threshold=0.99), text_kinds=roles)

    assert result.keep_indices == [0, 2, 3]


def test_role_separation_partitions_candidate_search() -> None:
    class RecordingIndex:
        def __init__(self) -> None:
            self.partition_sizes: list[int] = []

        def candidates(self, vectors: np.ndarray, _neighbors: int) -> list[tuple[int, int, float]]:
            self.partition_sizes.append(len(vectors))
            return [(0, 1, 1.0)] if len(vectors) > 1 else []

    backend = RecordingIndex()
    vectors = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    roles = [TextKind.COMMENT, TextKind.REPLY, TextKind.COMMENT, TextKind.REPLY]

    result = semantic_deduplicate(vectors, _config(), text_kinds=roles, candidate_index=backend)

    assert backend.partition_sizes == [2, 2]
    assert result.keep_indices == [0, 1]


def test_semantic_dedup_rejects_invalid_vectors_and_roles() -> None:
    with pytest.raises(ValueError, match="zero vectors"):
        semantic_deduplicate(np.zeros((2, 3), dtype=np.float32), _config())
    with pytest.raises(ValueError, match="text_kinds length"):
        semantic_deduplicate(np.ones((2, 3), dtype=np.float32), _config(), text_kinds=[TextKind.COMMENT])
    with pytest.raises(ValueError, match="exhaustive backend is limited"):
        semantic_deduplicate(
            np.ones((3, 2), dtype=np.float32),
            _config(exhaustive_max_records=2),
        )


def test_semantic_deduplication_artifacts_verify_embedding_alignment(tmp_path: Path) -> None:
    records_path = tmp_path / "development-clean.jsonl"
    records = [
        CleanedTextUnit(
            record_id=f"record-{index}",
            text=f"Original {index}",
            clean_text=f"Clean {index}",
            text_kind=TextKind.COMMENT,
            detected_language="english",
        )
        for index in range(3)
    ]
    records_path.write_text("".join(f"{record.model_dump_json()}\n" for record in records), encoding="utf-8")
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, np.asarray([[1.0, 0.0], [0.999, 0.01], [0.0, 1.0]], dtype=np.float32))
    embedding_manifest_path = tmp_path / "embedding-manifest.json"
    embedding_manifest = EmbeddingArtifactManifest(
        records_sha256=_sha256(records_path),
        embeddings_sha256=_sha256(embeddings_path),
        n_records=3,
        dimensions=2,
        model_name="test/model",
        normalized=True,
    )
    embedding_manifest_path.write_text(embedding_manifest.model_dump_json(), encoding="utf-8")

    manifest = run_semantic_deduplication(
        records_path,
        embeddings_path,
        embedding_manifest_path,
        tmp_path / "output",
        config=_config(),
    )

    assert manifest.result.n_removed == 1
    assert json.loads((tmp_path / "output" / "keep-indices.json").read_text(encoding="utf-8")) == [0, 2]
    restored = SemanticDeduplicationManifest.model_validate_json(
        (tmp_path / "output" / "semantic-deduplication-manifest.json").read_text(encoding="utf-8"),
    )
    assert restored == manifest
    assert restored.keep_indices_sha256 == _sha256(tmp_path / "output" / "keep-indices.json")
    assert restored.groups_sha256 == _sha256(tmp_path / "output" / "semantic-groups.jsonl")

    embeddings_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        run_semantic_deduplication(
            records_path,
            embeddings_path,
            embedding_manifest_path,
            tmp_path / "tampered-output",
            config=_config(),
        )


def test_semantic_deduplication_cli_uses_config(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    records = [
        CleanedTextUnit(
            record_id=f"record-{index}",
            text="Original",
            clean_text="Clean",
            text_kind=TextKind.COMMENT,
            detected_language="english",
        )
        for index in range(2)
    ]
    records_path.write_text("".join(f"{record.model_dump_json()}\n" for record in records), encoding="utf-8")
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    embedding_manifest_path = tmp_path / "embedding-manifest.json"
    embedding_manifest_path.write_text(
        EmbeddingArtifactManifest(
            records_sha256=_sha256(records_path),
            embeddings_sha256=_sha256(embeddings_path),
            n_records=2,
            dimensions=2,
            model_name="test/model",
            normalized=True,
        ).model_dump_json(),
        encoding="utf-8",
    )
    config_path = tmp_path / "dedup.json"
    config_path.write_text(_config().model_dump_json(), encoding="utf-8")
    output_dir = tmp_path / "output"

    exit_code = main(
        [
            str(records_path),
            str(embeddings_path),
            "--embedding-manifest",
            str(embedding_manifest_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert exit_code == 0
    assert json.loads((output_dir / "keep-indices.json").read_text(encoding="utf-8")) == [0]
