"""Checksum-bound preparation and persistence for restricted manual topic review."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.ml.evaluation import EvaluationManifest, ManualAnnotation, ManualReviewRecord

_MAX_REVIEW_NOTE_LENGTH = 2000


class _ReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ManualReviewBundle(_ReviewModel):
    """Contain verified review inputs and optional in-progress annotations."""

    run_id: str
    run_dir: str
    pipeline_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample: tuple[ManualReviewRecord, ...]
    annotations: tuple[ManualAnnotation, ...]


class ManualReviewSummary(_ReviewModel):
    """Summarize annotation coverage without treating judgments as ground truth."""

    sample_records: int = Field(ge=0)
    annotated_records: int = Field(ge=0)
    coverage: float = Field(ge=0, le=1)
    reviewers: tuple[str, ...]
    topic_match_share: float | None = Field(default=None, ge=0, le=1)
    topic_clear_share: float | None = Field(default=None, ge=0, le=1)
    business_relevant_share: float | None = Field(default=None, ge=0, le=1)
    reassignment_correct_share: float | None = Field(default=None, ge=0, le=1)
    sensitive_data_flags: int = Field(ge=0)
    merge_candidates: int = Field(ge=0)
    split_candidates: int = Field(ge=0)


class ReviewerAgreement(_ReviewModel):
    """Report observed agreement on overlapping binary judgments."""

    reviewer_pairs: int = Field(ge=0)
    overlapping_records: int = Field(ge=0)
    compared_judgments: int = Field(ge=0)
    matching_judgments: int = Field(ge=0)
    observed_agreement: float | None = Field(default=None, ge=0, le=1)
    interpretation: str = "observed agreement only; not chance-corrected reliability or ground truth"


class ManualAnnotationManifest(_ReviewModel):
    """Bind a restricted annotation file to its verified review sample."""

    schema_version: int = 1
    run_id: str
    pipeline_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotations_path: str
    annotations_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotations: int = Field(ge=0)
    classification: str = "restricted_manual_review"
    created_at: datetime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_run_path(run_dir: Path, raw_path: str, expected_sha256: str) -> Path:
    path = Path(raw_path).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        msg = f"manual-review input escapes the selected run: {path.name}"
        raise ValueError(msg)
    if not path.is_file() or path.is_symlink():
        msg = f"manual-review input is missing or unsafe: {path.name}"
        raise FileNotFoundError(msg)
    if _sha256(path) != expected_sha256:
        msg = f"manual-review input checksum mismatch: {path.name}"
        raise ValueError(msg)
    return path


def _load_sample(path: Path) -> list[ManualReviewRecord]:
    with path.open(encoding="utf-8") as source:
        return [ManualReviewRecord.model_validate_json(line) for line in source if line.strip()]


def _load_annotations(path: Path) -> list[ManualAnnotation]:
    with path.open(encoding="utf-8") as source:
        return [ManualAnnotation.model_validate_json(line) for line in source if line.strip()]


def validate_manual_annotations(
    sample: tuple[ManualReviewRecord, ...],
    annotations: list[ManualAnnotation] | tuple[ManualAnnotation, ...],
) -> tuple[ManualAnnotation, ...]:
    """Validate a pipeline-compatible, resumable annotation subset."""
    sample_by_index = {record.record_index: record for record in sample}
    indices = [annotation.record_index for annotation in annotations]
    if len(indices) != len(set(indices)):
        msg = "manual annotations contain duplicate record_index values"
        raise ValueError(msg)
    for annotation in annotations:
        if annotation.record_index not in sample_by_index:
            msg = f"manual annotation is outside the verified review sample: {annotation.record_index}"
            raise ValueError(msg)
        if not annotation.reviewer.strip():
            msg = f"manual annotation requires reviewer identity: {annotation.record_index}"
            raise ValueError(msg)
        if len(annotation.note) > _MAX_REVIEW_NOTE_LENGTH:
            msg = f"manual annotation note exceeds {_MAX_REVIEW_NOTE_LENGTH} characters: {annotation.record_index}"
            raise ValueError(msg)
        is_reassigned = sample_by_index[annotation.record_index].sample_kind == "reassigned"
        if is_reassigned != (annotation.reassignment_correct is not None):
            msg = f"reassignment_correct does not match sample kind: {annotation.record_index}"
            raise ValueError(msg)
    return tuple(sorted(annotations, key=lambda annotation: annotation.record_index))


def load_manual_review_bundle(run_dir: Path, annotations_path: Path | None = None) -> ManualReviewBundle:
    """Load the stage-12 sample and optional external annotations with checksums."""
    resolved = run_dir.resolve()
    pipeline_path = resolved / "run-manifest.json"
    evaluation_path = resolved / "12-evaluation/evaluation-manifest.json"
    if pipeline_path.is_symlink() or evaluation_path.is_symlink():
        msg = "manual-review manifests must not be symbolic links"
        raise ValueError(msg)
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    evaluation = EvaluationManifest.model_validate_json(evaluation_path.read_text(encoding="utf-8"))
    pipeline_sha256 = _sha256(pipeline_path)
    evaluation_sha256 = _sha256(evaluation_path)
    sample_path = _verified_run_path(
        resolved,
        evaluation.manual_review_sample_path,
        evaluation.manual_review_sample_sha256,
    )
    _verified_run_path(
        resolved,
        evaluation.manual_review_template_path,
        evaluation.manual_review_template_sha256,
    )
    sample = tuple(_load_sample(sample_path))
    sample_indices = [record.record_index for record in sample]
    if len(sample_indices) != len(set(sample_indices)):
        msg = "manual-review sample contains duplicate record_index values"
        raise ValueError(msg)
    loaded_annotations: list[ManualAnnotation] = []
    if annotations_path is not None and annotations_path.is_file():
        if annotations_path.is_symlink():
            msg = "manual annotations must not be a symbolic link"
            raise ValueError(msg)
        annotations_manifest_path = annotations_path.with_name("manual-annotations-manifest.json")
        if annotations_manifest_path.is_file():
            if annotations_manifest_path.is_symlink():
                msg = "manual annotation manifest must not be a symbolic link"
                raise ValueError(msg)
            annotation_manifest = ManualAnnotationManifest.model_validate_json(
                annotations_manifest_path.read_text(encoding="utf-8")
            )
            expected_binding = (
                str(pipeline["run_id"]),
                pipeline_sha256,
                evaluation_sha256,
                evaluation.manual_review_sample_sha256,
                _sha256(annotations_path),
            )
            actual_binding = (
                annotation_manifest.run_id,
                annotation_manifest.pipeline_manifest_sha256,
                annotation_manifest.evaluation_manifest_sha256,
                annotation_manifest.sample_sha256,
                annotation_manifest.annotations_sha256,
            )
            if actual_binding != expected_binding:
                msg = "manual annotations do not match the selected run or checksum"
                raise ValueError(msg)
        loaded_annotations = _load_annotations(annotations_path)
    annotations = validate_manual_annotations(sample, loaded_annotations)
    return ManualReviewBundle(
        run_id=str(pipeline["run_id"]),
        run_dir=str(resolved),
        pipeline_manifest_sha256=pipeline_sha256,
        evaluation_manifest_sha256=evaluation_sha256,
        sample_sha256=evaluation.manual_review_sample_sha256,
        sample=sample,
        annotations=annotations,
    )


def summarize_manual_annotations(bundle: ManualReviewBundle) -> ManualReviewSummary:
    """Summarize a partial or complete review without persisting source text."""
    annotations = bundle.annotations
    count = len(annotations)
    reassigned = [item for item in annotations if item.reassignment_correct is not None]
    return ManualReviewSummary(
        sample_records=len(bundle.sample),
        annotated_records=count,
        coverage=count / len(bundle.sample) if bundle.sample else 0,
        reviewers=tuple(sorted({item.reviewer for item in annotations})),
        topic_match_share=sum(item.topic_matches for item in annotations) / count if count else None,
        topic_clear_share=sum(item.topic_clear for item in annotations) / count if count else None,
        business_relevant_share=sum(item.business_relevant for item in annotations) / count if count else None,
        reassignment_correct_share=(
            sum(bool(item.reassignment_correct) for item in reassigned) / len(reassigned) if reassigned else None
        ),
        sensitive_data_flags=sum(item.contains_sensitive_data for item in annotations),
        merge_candidates=sum(item.merge_candidate for item in annotations),
        split_candidates=sum(item.split_candidate for item in annotations),
    )


def reviewer_agreement(annotation_sets: dict[str, tuple[ManualAnnotation, ...]]) -> ReviewerAgreement:
    """Calculate raw pairwise agreement for overlapping independently reviewed rows."""
    names = sorted(annotation_sets)
    compared = 0
    matching = 0
    overlapping: set[int] = set()
    pairs = 0
    for left_position, left_name in enumerate(names):
        left = {item.record_index: item for item in annotation_sets[left_name]}
        for right_name in names[left_position + 1 :]:
            pairs += 1
            right = {item.record_index: item for item in annotation_sets[right_name]}
            for record_index in sorted(left.keys() & right.keys()):
                overlapping.add(record_index)
                fields = ("topic_matches", "topic_clear", "business_relevant")
                for field in fields:
                    compared += 1
                    matching += getattr(left[record_index], field) == getattr(right[record_index], field)
                left_reassignment = left[record_index].reassignment_correct
                right_reassignment = right[record_index].reassignment_correct
                if left_reassignment is not None and right_reassignment is not None:
                    compared += 1
                    matching += left_reassignment == right_reassignment
    return ReviewerAgreement(
        reviewer_pairs=pairs,
        overlapping_records=len(overlapping),
        compared_judgments=compared,
        matching_judgments=matching,
        observed_agreement=matching / compared if compared else None,
    )


def save_manual_annotations(
    bundle: ManualReviewBundle,
    annotations: list[ManualAnnotation] | tuple[ManualAnnotation, ...],
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> ManualAnnotationManifest:
    """Atomically save validated restricted annotations outside the immutable run."""
    target = output_dir.resolve()
    if target.is_relative_to(Path(bundle.run_dir).resolve()):
        msg = "manual annotations must be stored outside the immutable run"
        raise ValueError(msg)
    validated = validate_manual_annotations(bundle.sample, annotations)
    annotations_path = target / "manual-annotations.jsonl"
    manifest_path = target / "manual-annotations-manifest.json"
    if not overwrite and (annotations_path.exists() or manifest_path.exists()):
        msg = f"manual annotations already exist: {target}"
        raise FileExistsError(msg)
    target.mkdir(parents=True, exist_ok=True)
    annotations_tmp = annotations_path.with_name(f".{annotations_path.name}.tmp")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with annotations_tmp.open("w", encoding="utf-8") as output:
        for annotation in validated:
            output.write(f"{annotation.model_dump_json()}\n")
    annotations_tmp.replace(annotations_path)
    manifest = ManualAnnotationManifest(
        run_id=bundle.run_id,
        pipeline_manifest_sha256=bundle.pipeline_manifest_sha256,
        evaluation_manifest_sha256=bundle.evaluation_manifest_sha256,
        sample_sha256=bundle.sample_sha256,
        annotations_path=str(annotations_path),
        annotations_sha256=_sha256(annotations_path),
        annotations=len(validated),
        created_at=datetime.now(UTC),
    )
    manifest_tmp.write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    return manifest
