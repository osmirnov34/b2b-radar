import json
import os
from pathlib import Path

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.infrastructure.analytics_db import (
    ANALYTICS_SCHEMA,
    AnalyticsBase,
    AnalyticsProblemAssessmentModel,
    AnalyticsProblemRunModel,
)
from src.operations.feedback_loading import load_feedback_snapshot
from src.operations.feedback_preparation import prepare_yandex_feedback
from src.operations.problem_assessment import assess_feedback_problems
from src.operations.problem_loading import ProblemLoadError, load_problem_assessments, verify_problem_artifact

PROJECT_ROOT = Path(__file__).parents[1]
TAXONOMY = PROJECT_ROOT / "configs/dental-problem-taxonomy.v1.json"
RULES = PROJECT_ROOT / "configs/dental-problem-rules.v1.json"


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "reviews.jsonl"
    source.write_text(
        json.dumps(
            {
                "review_id": "7995cc14c8b92196",
                "review_text": "Долго ждала в очереди, администратор грубил.",
                "rating": 1,
                "published_at": "2025-04-03",
                "org_id": "229248496819",
                "org_name": "Стоматология",
                "org_url": "https://yandex.ru/maps/org/229248496819/reviews/",
                "search_query": "стоматология",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    feedback = tmp_path / "feedback.jsonl"
    feedback_manifest = tmp_path / "feedback.manifest.json"
    prepare_yandex_feedback(source, feedback, feedback_manifest)
    assessments = tmp_path / "assessments.jsonl"
    assess_feedback_problems(feedback, feedback_manifest, TAXONOMY, RULES, assessments)
    return feedback, feedback_manifest, assessments, assessments.with_suffix(".jsonl.manifest.json")


def test_verify_problem_artifact_reconciles_manifest(tmp_path: Path) -> None:
    _, _, assessments, manifest = _artifacts(tmp_path)

    verified = verify_problem_artifact(assessments, manifest)

    assert len(verified.assessments) == verified.manifest.assessments
    assert {item.feedback_id for item in verified.assessments} == {"yandex_maps:7995cc14c8b92196"}


def test_verify_problem_artifact_rejects_changed_bytes_without_text(tmp_path: Path) -> None:
    _, _, assessments, manifest = _artifacts(tmp_path)
    private_text = "Секретный отзыв"
    assessments.write_text(assessments.read_text(encoding="utf-8") + private_text, encoding="utf-8")

    with pytest.raises(ProblemLoadError) as captured:
        verify_problem_artifact(assessments, manifest)

    assert str(captured.value) == "problem assessment artifact checksum does not match manifest"
    assert private_text not in str(captured.value)


def test_verify_problem_artifact_rejects_changed_taxonomy(tmp_path: Path) -> None:
    _, _, assessments, manifest = _artifacts(tmp_path)
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_bytes(TAXONOMY.read_bytes() + b"\n")

    with pytest.raises(ProblemLoadError, match="taxonomy is missing or does not match"):
        verify_problem_artifact(assessments, manifest, taxonomy_path=taxonomy)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_problem_loading_is_transactional_and_idempotent(tmp_path: Path) -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip("TEST_DATABASE_URL is not configured")
    feedback, feedback_manifest, assessments, assessment_manifest = _artifacts(tmp_path)
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(f"DROP SCHEMA IF EXISTS {ANALYTICS_SCHEMA} CASCADE")
            await connection.exec_driver_sql(f"CREATE SCHEMA {ANALYTICS_SCHEMA}")
            await connection.run_sync(AnalyticsBase.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            await load_feedback_snapshot(session, feedback, feedback_manifest)
            first = await load_problem_assessments(session, assessments, assessment_manifest, batch_size=1)
            second = await load_problem_assessments(session, assessments, assessment_manifest, batch_size=1)
            run_count = await session.scalar(select(func.count()).select_from(AnalyticsProblemRunModel))
            assessment_count = await session.scalar(
                select(func.count()).select_from(AnalyticsProblemAssessmentModel),
            )

        assert first.already_loaded is False
        assert second.already_loaded is True
        assert second.run_id == first.run_id
        assert first.categories == 10
        assert run_count == 1
        assert assessment_count == first.assessments
        async with session_factory() as session:
            estimates = dict(
                (
                    await session.execute(
                        text(
                            "SELECT relname, reltuples FROM pg_class WHERE oid IN "
                            "('analytics.problem_runs'::regclass, 'analytics.problem_categories'::regclass, "
                            "'analytics.problem_assessments'::regclass)"
                        )
                    )
                )
                .tuples()
                .all()
            )
        assert estimates == {
            "problem_runs": 1,
            "problem_categories": first.categories,
            "problem_assessments": first.assessments,
        }
    finally:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(f"DROP SCHEMA IF EXISTS {ANALYTICS_SCHEMA} CASCADE")
        await engine.dispose()
