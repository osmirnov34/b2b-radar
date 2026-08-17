import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from scripts.build_corpus import main
from src.ml.cleaning_dataset import DatasetCleaningManifest, DatasetCleaningStats
from src.ml.config import CleaningConfig, DeduplicationConfig, EmbeddingConfig
from src.ml.corpus import CorpusManifest, CorpusRecord, build_final_corpus
from src.ml.embeddings import generate_embeddings
from src.ml.semantic_deduplication import SemanticDeduplicationManifest, run_semantic_deduplication


class FixedEncoder:
    device = "cpu"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = {
            "Private alpha issue": [1.0, 0.0, 0.0],
            "Private alpha duplicate": [1.0, 0.0, 0.0],
            "Private gamma issue": [0.0, 1.0, 0.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_inputs(tmp_path: Path) -> dict[str, Path]:
    records = tmp_path / "development-clean.jsonl"
    rows = [
        {
            "record_id": f"record-{index}",
            "text": text,
            "clean_text": text,
            "text_kind": "comment",
            "detected_language": "en",
            "video_id": "private-video-a" if index < 2 else "private-video-b",
            "duplicate_count": index + 1,
        }
        for index, text in enumerate(
            ("Private alpha issue", "Private alpha duplicate", "Private gamma issue"),
        )
    ]
    records.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    cleaning_manifest_path = tmp_path / "cleaning-manifest.json"
    cleaning_manifest = DatasetCleaningManifest(
        source_path="development.jsonl",
        source_sha256="0" * 64,
        split_manifest_path="split-manifest.json",
        config=CleaningConfig(),
        output_path=str(records),
        output_sha256=_sha256(records),
        decisions_sha256="1" * 64,
        stats=DatasetCleaningStats(
            input_rows=3,
            input_text_units=3,
            input_comments=3,
            input_replies=0,
            output_text_units=3,
            output_comments=3,
            output_replies=0,
            removed_by_reason={},
            detected_languages={"en": 3},
            duplicate_groups=0,
            largest_duplicate_group=0,
        ),
        warnings=[],
        created_at=datetime.now(UTC),
    )
    cleaning_manifest_path.write_text(cleaning_manifest.model_dump_json(), encoding="utf-8")
    embedding_dir = tmp_path / "embeddings"
    generate_embeddings(
        records,
        embedding_dir,
        config=EmbeddingConfig(model_name="fake/model", batch_size=2),
        encoder=FixedEncoder(),
    )
    deduplication_dir = tmp_path / "semantic-deduplication"
    run_semantic_deduplication(
        records,
        embedding_dir / "embeddings.npy",
        embedding_dir / "embedding-manifest.json",
        deduplication_dir,
        config=DeduplicationConfig(backend="exhaustive", threshold=0.99),
    )
    return {
        "records": records,
        "cleaning_manifest": cleaning_manifest_path,
        "embeddings": embedding_dir / "embeddings.npy",
        "embedding_manifest": embedding_dir / "embedding-manifest.json",
        "keep_indices": deduplication_dir / "keep-indices.json",
        "groups": deduplication_dir / "semantic-groups.jsonl",
        "deduplication_manifest": deduplication_dir / "semantic-deduplication-manifest.json",
    }


def _build(paths: dict[str, Path], output: Path, *, force: bool = False) -> CorpusManifest:
    return build_final_corpus(
        paths["records"],
        paths["embeddings"],
        paths["keep_indices"],
        cleaning_manifest_path=paths["cleaning_manifest"],
        embedding_manifest_path=paths["embedding_manifest"],
        groups_path=paths["groups"],
        deduplication_manifest_path=paths["deduplication_manifest"],
        output_dir=output,
        force=force,
    )


def test_build_final_corpus_preserves_alignment_and_provenance(tmp_path: Path) -> None:
    paths = _build_inputs(tmp_path)
    output = tmp_path / "corpus"

    manifest = _build(paths, output)

    records = [
        CorpusRecord.model_validate_json(line)
        for line in (output / "final-corpus.jsonl").read_text().splitlines()
    ]
    ids = [json.loads(line) for line in (output / "final-record-ids.jsonl").read_text().splitlines()]
    source_embeddings = np.load(paths["embeddings"], allow_pickle=False)
    final_embeddings = np.load(output / "final-embeddings.npy", allow_pickle=False)
    assert [record.cleaned_record_index for record in records] == [0, 2]
    assert [record.record_id for record in records] == ids == ["record-0", "record-2"]
    assert records[0].semantic_duplicate_count == 1
    assert records[0].duplicate_count == 1
    assert np.array_equal(final_embeddings, source_embeddings[[0, 2]])
    assert manifest.stats.input_records == 3
    assert manifest.stats.output_records == 2
    assert manifest.stats.removed_semantic_duplicates == 1
    assert CorpusManifest.model_validate_json((output / "corpus-manifest.json").read_text()) == manifest
    report = (output / "corpus-report.md").read_text()
    assert "Private alpha" not in report
    assert "private-video" not in report


def test_build_final_corpus_rejects_unordered_or_inconsistent_indices(tmp_path: Path) -> None:
    paths = _build_inputs(tmp_path)
    paths["keep_indices"].write_text("[2, 0]\n", encoding="utf-8")
    manifest = SemanticDeduplicationManifest.model_validate_json(paths["deduplication_manifest"].read_text())
    updated = manifest.model_copy(update={"keep_indices_sha256": _sha256(paths["keep_indices"])})
    paths["deduplication_manifest"].write_text(updated.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="unique and strictly increasing"):
        _build(paths, tmp_path / "corpus")


def test_build_final_corpus_rejects_empty_keep_set_without_representatives(tmp_path: Path) -> None:
    paths = _build_inputs(tmp_path)
    paths["keep_indices"].write_text("[]\n", encoding="utf-8")
    manifest = SemanticDeduplicationManifest.model_validate_json(paths["deduplication_manifest"].read_text())
    result = manifest.result.model_copy(update={"n_kept": 0, "n_removed": 3})
    updated = manifest.model_copy(
        update={"keep_indices_sha256": _sha256(paths["keep_indices"]), "result": result},
    )
    paths["deduplication_manifest"].write_text(updated.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match=r"representative .* is not a valid kept index"):
        _build(paths, tmp_path / "corpus")


def test_build_final_corpus_rejects_tampered_record_alignment(tmp_path: Path) -> None:
    paths = _build_inputs(tmp_path)
    embedding_manifest = json.loads(paths["embedding_manifest"].read_text())
    record_ids = Path(embedding_manifest["record_ids_path"])
    record_ids.write_text('"record-1"\n"record-0"\n"record-2"\n', encoding="utf-8")
    embedding_manifest["record_ids_sha256"] = _sha256(record_ids)
    paths["embedding_manifest"].write_text(json.dumps(embedding_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="alignment mismatch"):
        _build(paths, tmp_path / "corpus")


def test_build_final_corpus_protects_outputs_and_force_replaces_them(tmp_path: Path) -> None:
    paths = _build_inputs(tmp_path)
    output = tmp_path / "corpus"
    _build(paths, output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _build(paths, output)
    assert _build(paths, output, force=True).stats.output_records == 2


def test_build_corpus_cli_runs_complete_fixture_chain(tmp_path: Path) -> None:
    paths = _build_inputs(tmp_path)
    output = tmp_path / "corpus"
    exit_code = main(
        [
            str(paths["records"]),
            str(paths["embeddings"]),
            str(paths["keep_indices"]),
            "--cleaning-manifest",
            str(paths["cleaning_manifest"]),
            "--embedding-manifest",
            str(paths["embedding_manifest"]),
            "--groups",
            str(paths["groups"]),
            "--deduplication-manifest",
            str(paths["deduplication_manifest"]),
            "--output-dir",
            str(output),
        ],
    )
    assert exit_code == 0
    assert (output / "corpus-manifest.json").is_file()
