import tomllib
from pathlib import Path

import pytest

from src.operations.ml_pipeline import _python_version_supported

PROJECT_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("version", [(3, 11), (3, 12), (3, 13)])
def test_ml_pipeline_accepts_supported_python_versions(version: tuple[int, int]) -> None:
    assert _python_version_supported(version)


@pytest.mark.parametrize("version", [(3, 10), (3, 14), (4, 0)])
def test_ml_pipeline_rejects_unsupported_python_versions(version: tuple[int, int]) -> None:
    assert not _python_version_supported(version)


def test_project_metadata_matches_pipeline_python_range() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["requires-python"] == ">=3.11,<3.14"
