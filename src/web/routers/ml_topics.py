"""Read-only HTTP API for safe public ML exports."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from src.web.ml_snapshot import (
    MLSnapshot,
    MLSnapshotRepository,
    PublicAssignment,
    PublicQuality,
    PublicTopic,
    SnapshotUnavailableError,
)

router = APIRouter(prefix="/api", tags=["ml-topics"])


class _ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TopicListResponse(_ResponseModel):
    items: list[PublicTopic]
    total: int = Field(ge=0)


class TopicDetailResponse(_ResponseModel):
    topic: PublicTopic


class AssignmentListResponse(_ResponseModel):
    items: list[PublicAssignment]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)


class QualityResponse(_ResponseModel):
    quality: PublicQuality
    reload_error: str | None = None


async def get_ml_repository(request: Request) -> MLSnapshotRepository:
    repository = getattr(request.app.state, "ml_repository", None)
    if not isinstance(repository, MLSnapshotRepository):
        raise HTTPException(status_code=503, detail="ML results are not configured")
    return repository


RepositoryDep = Annotated[MLSnapshotRepository, Depends(get_ml_repository)]


def _snapshot(repository: MLSnapshotRepository) -> MLSnapshot:
    try:
        return repository.get_snapshot()
    except SnapshotUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/topics")
async def list_topics(repository: RepositoryDep) -> TopicListResponse:
    snapshot = _snapshot(repository)
    return TopicListResponse(items=list(snapshot.topics), total=len(snapshot.topics))


@router.get("/topics/{topic_id}")
async def topic_detail(topic_id: int, repository: RepositoryDep) -> TopicDetailResponse:
    _snapshot(repository)
    topic = repository.topic(topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")
    return TopicDetailResponse(topic=topic)


@router.get("/topics/{topic_id}/assignments")
async def topic_assignments(
    topic_id: int,
    repository: RepositoryDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> AssignmentListResponse:
    snapshot = _snapshot(repository)
    if topic_id not in snapshot.assignments_by_topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    assignments = snapshot.assignments_by_topic[topic_id]
    return AssignmentListResponse(
        items=list(assignments[offset : offset + limit]),
        total=len(assignments),
        offset=offset,
        limit=limit,
    )


@router.get("/ml/outliers")
async def outlier_assignments(
    repository: RepositoryDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> AssignmentListResponse:
    assignments = _snapshot(repository).assignments_by_topic[None]
    return AssignmentListResponse(
        items=list(assignments[offset : offset + limit]),
        total=len(assignments),
        offset=offset,
        limit=limit,
    )


@router.get("/ml/quality")
async def ml_quality(repository: RepositoryDep) -> QualityResponse:
    snapshot = _snapshot(repository)
    return QualityResponse(quality=snapshot.quality, reload_error=repository.last_error)
