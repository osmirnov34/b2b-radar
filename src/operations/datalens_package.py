"""Build a deterministic, credential-free assembly package for DataLens."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SPEC_PATH = Path(__file__).resolve().parents[2] / "configs/datalens-dashboard.v1.json"
_SOURCES = frozenset({"organization_overview", "organization_monthly", "category_overview", "category_monthly"})


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceSpec(_Model):
    grain: tuple[str, ...]


class FilterSpec(_Model):
    field: str
    datasets: tuple[str, ...]


class ChartSpec(_Model):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    title: str
    dataset: str
    type: Literal["indicator", "table", "bar", "line"]
    dimensions: tuple[str, ...]
    metrics: tuple[str, ...] = Field(min_length=1)


class DashboardSpec(_Model):
    """Assembly specification; this is not a native DataLens import format."""

    schema_version: Literal[1]
    title: str
    minimum_migration: str
    datasets: dict[str, SourceSpec]
    metrics: dict[str, str]
    filters: tuple[FilterSpec, ...]
    charts: tuple[ChartSpec, ...]

    @model_validator(mode="after")
    def validate_references(self) -> DashboardSpec:
        if set(self.datasets) != _SOURCES:
            msg = "dashboard must use exactly the four supported aggregate sources"
            raise ValueError(msg)
        ids = [chart.id for chart in self.charts]
        if len(ids) != len(set(ids)):
            msg = "dashboard chart IDs must be unique"
            raise ValueError(msg)
        for chart in self.charts:
            if chart.dataset not in self.datasets or set(chart.metrics) - self.metrics.keys():
                msg = "chart references an unknown dataset or metric"
                raise ValueError(msg)
            if (
                chart.dataset.startswith("category")
                and any(metric.endswith("_share") for metric in chart.metrics)
                and "category_id" not in chart.dimensions
            ):
                msg = "category shares require category_id in chart dimensions"
                raise ValueError(msg)
        if any(set(selector.datasets) - self.datasets.keys() for selector in self.filters):
            msg = "filter references an unknown dataset"
            raise ValueError(msg)
        return self


def dataset_sql(source: str, run_id: int) -> str:
    """Select exactly one assessment run from an allowlisted aggregate view."""
    if source not in _SOURCES or type(run_id) is not int or run_id < 1:
        msg = "expected an aggregate source and a positive integer run_id"
        raise ValueError(msg)
    return f"SELECT * FROM analytics_bi.{source} WHERE run_id = {run_id}\n"  # noqa: S608 -- allowlist + strict integer.


def build_datalens_package(output_dir: Path, run_id: int) -> dict[str, object]:
    """Publish a new local assembly directory; never change DataLens or a database."""
    dataset_sql("organization_overview", run_id)
    if output_dir.exists():
        msg = f"refusing to overwrite DataLens package: {output_dir}"
        raise FileExistsError(msg)
    spec = DashboardSpec.model_validate_json(_SPEC_PATH.read_text(encoding="utf-8"))
    files = {f"{source}.sql": dataset_sql(source, run_id) for source in sorted(spec.datasets)}
    files["dashboard.json"] = spec.model_dump_json(indent=2) + "\n"
    files["README.md"] = (
        f"# {spec.title}\n\nRun ID: {run_id}. Minimum migration: {spec.minimum_migration}.\n\n"
        "SQL files are dataset subqueries. dashboard.json is an assembly specification, not a native import file.\n"
        "Follow docs/datalens-dashboard.md in the repository.\n"
        "This package contains no database credentials or reviews.\n"
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "minimum_migration": spec.minimum_migration,
        "files_sha256": {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()},
        "status": "assembly_package_only",
    }
    files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".datalens-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        for name, content in files.items():
            (staging / name).write_text(content, encoding="utf-8")
        staging.rename(output_dir)
    return manifest
