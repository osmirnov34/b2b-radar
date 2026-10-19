"""Safety checks for the isolated local demonstration setup."""

import hashlib
from pathlib import Path

import pytest
from dotenv import dotenv_values

from src.operations.analytics_demo import _verify_artifacts, initialize_analytics_demo, load_demo_environment
from src.operations.analytics_pipeline import AnalyticsStage


def test_demo_initialization_is_private_and_never_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "reviews.jsonl"
    source.write_text("{}\n")
    directory = tmp_path / "demo"
    initialize_analytics_demo(source, directory)
    environment = directory / "environment.env"
    original = environment.read_bytes()
    values = dotenv_values(environment)
    assert environment.stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700
    assert values["POSTGRES_HOST"] == "127.0.0.1"
    assert len({values[key] for key in ("POSTGRES_PASSWORD", "ANALYTICS_API_TOKEN", "ANALYTICS_PRIVATE_TOKEN")}) == 3
    with pytest.raises(FileExistsError):
        initialize_analytics_demo(source, directory)
    assert environment.read_bytes() == original


def test_demo_requires_source_before_creating_directory(tmp_path: Path) -> None:
    directory = tmp_path / "demo"
    with pytest.raises(FileNotFoundError, match="source file is missing"):
        initialize_analytics_demo(tmp_path / "missing", directory)
    assert not directory.exists()


def test_demo_refuses_nonlocal_database_before_loading_credentials(tmp_path: Path) -> None:
    (tmp_path / "environment.env").write_text(
        "POSTGRES_HOST=remote.example\nPOSTGRES_USER=analytics_demo\nPOSTGRES_DB=analytics_demo\n"
    )
    with pytest.raises(ValueError, match="isolated local"):
        load_demo_environment(tmp_path)


def test_acceptance_verifies_every_checkpoint_and_rejects_tampering(tmp_path: Path) -> None:
    completed = {}
    for index, stage in enumerate(AnalyticsStage, start=1):
        directory = tmp_path / f"{index:02d}-{stage.value}"
        directory.mkdir()
        (directory / "artifact").write_bytes(b"aggregate")
        completed[stage.value] = {"artifact": hashlib.sha256(b"aggregate").hexdigest()}
    state = {"completed": completed}
    _verify_artifacts(tmp_path, state)
    (tmp_path / "08-pdf/artifact").write_bytes(b"changed")
    with pytest.raises(ValueError, match="do not match"):
        _verify_artifacts(tmp_path, state)
