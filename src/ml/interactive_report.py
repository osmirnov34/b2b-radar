"""Build one portable, checksum-bound HTML report from verified ML artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ml.reporting import (
    AnalysisArtifacts,
    ProblemPriorityConfig,
    ProblemSignalConfig,
    TemporalAnalysisConfig,
    VideoConcentrationConfig,
    analyze_topic_temporal_trends,
    assess_topic_problem_signals,
    assess_video_concentration,
    build_cluster_cards,
    build_data_lineage,
    rank_problem_topics,
    representative_comments,
    topic_summary_rows,
)

if TYPE_CHECKING:
    from pathlib import Path

_MAX_TITLE_LENGTH = 200


class _InteractiveReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InteractiveReportConfig(_InteractiveReportModel):
    """Configure report contents without changing source pipeline artifacts."""

    schema_version: int = 1
    title: str = "B2B Radar — ML analysis"
    include_private_examples: bool = False
    representatives_per_topic: int = Field(default=3, ge=1, le=20)
    problem_signals: ProblemSignalConfig = Field(default_factory=ProblemSignalConfig)
    problem_priority: ProblemPriorityConfig = Field(default_factory=ProblemPriorityConfig)
    video_concentration: VideoConcentrationConfig = Field(default_factory=VideoConcentrationConfig)
    temporal_analysis: TemporalAnalysisConfig = Field(default_factory=TemporalAnalysisConfig)

    @model_validator(mode="after")
    def validate_title(self) -> InteractiveReportConfig:
        if not self.title.strip() or len(self.title) > _MAX_TITLE_LENGTH:
            msg = "interactive report title must contain 1 to 200 characters"
            raise ValueError(msg)
        return self


class InteractiveReportManifest(_InteractiveReportModel):
    """Bind a generated HTML document to one immutable pipeline run."""

    report_schema_version: int = 1
    run_id: str
    pipeline_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: InteractiveReportConfig
    analyzed_at: datetime
    report_path: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    topics: int = Field(ge=0)
    examples: int = Field(ge=0)
    classification: str
    private_text_included: bool
    generated_at: datetime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_for_script(value: object) -> str:
    """Serialize JSON that cannot terminate its containing HTML script element."""
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _model_json(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _build_payload(
    artifacts: AnalysisArtifacts,
    config: InteractiveReportConfig,
    analyzed_at: datetime,
) -> dict[str, Any]:
    lineage = build_data_lineage(artifacts)
    cards = build_cluster_cards(artifacts)
    summaries = topic_summary_rows(artifacts)
    assessments = assess_topic_problem_signals(artifacts, config=config.problem_signals)
    priorities = rank_problem_topics(assessments, config=config.problem_priority)
    concentrations = assess_video_concentration(
        artifacts,
        signal_config=config.problem_signals,
        config=config.video_concentration,
    )
    trends = analyze_topic_temporal_trends(
        artifacts,
        signal_config=config.problem_signals,
        config=config.temporal_analysis,
        analyzed_at=analyzed_at,
    )
    examples = (
        representative_comments(artifacts, n=config.representatives_per_topic)
        if config.include_private_examples
        else []
    )
    by_topic: dict[int, dict[str, Any]] = {
        row.topic_id: {"summary": _model_json(row), "examples": []} for row in summaries
    }
    for key, rows in (
        ("card", cards),
        ("problem", assessments),
        ("concentration", concentrations),
        ("trend", trends),
    ):
        for row in rows:
            by_topic[row.topic_id][key] = _model_json(row)
    for priority in priorities:
        by_topic[priority.topic_id]["priority"] = _model_json(priority)
    for example in examples:
        by_topic[example.topic_id]["examples"].append(_model_json(example))
    return {
        "meta": {
            "run_id": artifacts.run_id,
            "analyzed_at": analyzed_at.isoformat(),
            "classification": "restricted" if config.include_private_examples else "aggregate_internal",
            "private_text_included": config.include_private_examples,
        },
        "summary": _model_json(artifacts.summary),
        "lineage": _model_json(lineage),
        "topics": [by_topic[topic_id] for topic_id in sorted(by_topic)],
    }


def _render_html(payload: dict[str, Any], title: str) -> str:
    data = _json_for_script(payload)
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title><style>
:root{{--bg:#f4f7fb;--card:#fff;--ink:#172033;--muted:#64748b;--accent:#2563eb;--warn:#b45309;--line:#dbe3ef}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
header{{background:#101827;color:white;padding:24px}}header h1{{margin:0 0 8px}}main{{max-width:1400px;margin:auto;padding:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}}
.metric b{{display:block;font-size:24px}}.muted{{color:var(--muted)}}input,select{{padding:10px;border:1px solid var(--line);border-radius:8px;margin:0 8px 12px 0}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:white}}
.bar{{height:10px;background:#dbeafe;border-radius:8px;overflow:hidden}}.bar i{{display:block;height:100%;background:var(--accent)}}.chart-row{{display:grid;grid-template-columns:minmax(140px,2fr) 5fr 70px;gap:10px;align-items:center;margin:8px 0}}.months{{display:flex;align-items:end;gap:2px;height:42px}}.months i{{min-width:5px;background:#60a5fa;border-radius:2px 2px 0 0}}details{{border-top:1px solid var(--line);padding:10px 0}}summary{{cursor:pointer;font-weight:650}}
.badge{{display:inline-block;padding:2px 7px;border-radius:12px;background:#e2e8f0}}.warning{{color:var(--warn)}}a{{color:var(--accent)}}.hidden{{display:none}}
</style></head><body><header><h1>{safe_title}</h1><div id="meta"></div></header><main>
<section id="metrics" class="grid"></section>
<section class="card"><h2>Поток данных</h2><div id="lineage"></div></section>
<section class="card"><h2>Крупнейшие темы</h2><div id="topic-chart"></div></section>
<section class="card"><h2>Темы</h2><input id="search" placeholder="Поиск по названию или ключевым словам"><select id="status"><option value="">Все статусы</option><option>problem_candidate</option><option>uncertain</option><option>topic_only</option></select><div id="topics"></div></section>
<section class="card"><h2>Ограничения интерпретации</h2><p>Кластеры являются исследовательскими темами. Приоритет не означает серьёзность проблемы; временная динамика не доказывает рыночный тренд; концентрация по видео требует ручной проверки.</p><div id="warnings"></div></section>
</main><script id="report-data" type="application/json">{data}</script><script>
const D=JSON.parse(document.getElementById('report-data').textContent), esc=v=>String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
document.getElementById('meta').textContent=`Run: ${{D.meta.run_id}} · ${{D.meta.classification}} · ${{D.meta.analyzed_at}}`;
const metric=(n,v,s='')=>`<div class="card metric"><span class="muted">${{esc(n)}}</span><b>${{esc(v)}}</b><small>${{esc(s)}}</small></div>`;
document.getElementById('metrics').innerHTML=metric('Текстов',D.summary.records)+metric('Тем',D.summary.topics)+metric('Выбросов',(100*D.summary.outlier_share).toFixed(1)+'%')+metric('Средняя уверенность',D.summary.mean_confidence.toFixed(3))+metric('Статус',D.summary.pipeline_status);
document.getElementById('lineage').innerHTML=`<table><thead><tr><th>Метрика</th><th>Количество</th><th>Scope</th><th>Определение</th></tr></thead><tbody>${{D.lineage.metrics.map(x=>`<tr><td>${{esc(x.label)}}</td><td>${{x.records}}</td><td>${{esc(x.scope)}}</td><td>${{esc(x.description)}}</td></tr>`).join('')}}</tbody></table>`;
const largest=[...D.topics].sort((a,b)=>b.summary.records-a.summary.records).slice(0,20),maxRecords=Math.max(1,...largest.map(t=>t.summary.records));document.getElementById('topic-chart').innerHTML=largest.map(t=>`<div class="chart-row"><span>#${{t.summary.topic_id}} ${{esc(t.summary.name)}}</span><div class="bar"><i style="width:${{100*t.summary.records/maxRecords}}%"></i></div><b>${{t.summary.records}}</b></div>`).join('');
const monthBars=tr=>{{const points=tr.monthly_activity||[],max=Math.max(1,...points.map(x=>x.records));return points.length?`<div class="months" aria-label="Помесячная активность">${{points.map(x=>`<i title="${{esc(x.month)}}: ${{x.records}}" style="height:${{Math.max(2,40*x.records/max)}}px"></i>`).join('')}}</div>`:''}};
function render(){{const q=document.getElementById('search').value.toLowerCase(),st=document.getElementById('status').value;const rows=D.topics.filter(t=>{{const s=t.summary,p=t.problem;return (!q||(s.name+' '+s.keywords.join(' ')).toLowerCase().includes(q))&&(!st||p.status===st)}});document.getElementById('topics').innerHTML=rows.map(t=>{{const s=t.summary,c=t.card,p=t.problem,v=t.concentration,tr=t.trend,pr=t.priority,ex=t.examples||[];return `<details><summary>#${{s.topic_id}} ${{esc(s.name)}} · ${{s.records}} · <span class="badge">${{esc(p.status)}}</span></summary><div class="grid">${{metric('Доля корпуса',(100*s.corpus_share).toFixed(2)+'%')}}${{metric('HDBSCAN',s.mean_probability.toFixed(3))}}${{metric('Проблемные сигналы',(100*p.signal_share).toFixed(1)+'%')}}${{metric('Приоритет',pr?pr.priority_score.toFixed(1):'—')}}${{metric('Видео',s.unique_videos)}}${{metric('Концентрация',v.topic_status)}}</div><p><b>Ключевые слова:</b> ${{esc(s.keywords.join(', '))}}</p><p><b>Динамика:</b> ${{esc(tr.status)}}; последний завершённый месяц: ${{esc(tr.latest_month||'нет данных')}}; изменение: ${{tr.relative_change==null?'—':(100*tr.relative_change).toFixed(1)+'%'}}</p>${{monthBars(tr)}}<p><b>Состав:</b> комментарии ${{c.comments}}, ответы ${{c.replies}}, переназначенные выбросы ${{c.reassigned_outliers}}</p>${{ex.length?`<h4>Репрезентативные примеры (restricted)</h4>${{ex.map(e=>`<blockquote>${{esc(e.text)}}<br><small>similarity=${{e.centroid_similarity.toFixed(3)}} · <a href="${{esc(e.video_url)}}" target="_blank" rel="noopener noreferrer">источник</a></small></blockquote>`).join('')}}`:''}}</details>`}}).join('')||'<p>Нет тем по фильтру.</p>'}}
document.getElementById('search').addEventListener('input',render);document.getElementById('status').addEventListener('change',render);render();
document.getElementById('warnings').innerHTML=(D.summary.warnings.length?D.summary.warnings:['Нет предупреждений']).map(x=>`<p class="warning">${{esc(x)}}</p>`).join('');
</script></body></html>"""


def write_interactive_report(
    artifacts: AnalysisArtifacts,
    output_dir: Path,
    *,
    config: InteractiveReportConfig | None = None,
    analyzed_at: datetime | None = None,
    overwrite: bool = False,
) -> InteractiveReportManifest:
    """Atomically write one self-contained report outside the immutable run."""
    active_config = config or InteractiveReportConfig()
    active_analyzed_at = analyzed_at or datetime.now(UTC)
    if active_analyzed_at.tzinfo is None:
        msg = "interactive report analyzed_at must be timezone-aware"
        raise ValueError(msg)
    target = output_dir.resolve()
    if output_dir.is_symlink():
        msg = "interactive report output directory must not be a symbolic link"
        raise ValueError(msg)
    if target.is_relative_to(artifacts.run_dir.resolve()):
        msg = "interactive report must be stored outside the immutable run"
        raise ValueError(msg)
    report_path = target / "interactive-report.html"
    manifest_path = target / "interactive-report-manifest.json"
    if report_path.is_symlink() or manifest_path.is_symlink():
        msg = "interactive report output files must not be symbolic links"
        raise ValueError(msg)
    if not overwrite and (report_path.exists() or manifest_path.exists()):
        msg = f"interactive report already exists: {target}"
        raise FileExistsError(msg)
    payload = _build_payload(artifacts, active_config, active_analyzed_at)
    html = _render_html(payload, active_config.title)
    target.mkdir(parents=True, exist_ok=True)
    report_tmp = report_path.with_name(f".{report_path.name}.tmp")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    if report_tmp.is_symlink() or manifest_tmp.is_symlink():
        msg = "interactive report temporary files must not be symbolic links"
        raise ValueError(msg)
    report_tmp.write_text(html, encoding="utf-8")
    report_tmp.replace(report_path)
    example_count = sum(len(topic["examples"]) for topic in payload["topics"])
    manifest = InteractiveReportManifest(
        run_id=artifacts.run_id,
        pipeline_manifest_sha256=artifacts.pipeline_manifest_sha256,
        config=active_config,
        analyzed_at=active_analyzed_at,
        report_path=str(report_path),
        report_sha256=_sha256(report_path),
        topics=len(payload["topics"]),
        examples=example_count,
        classification=payload["meta"]["classification"],
        private_text_included=active_config.include_private_examples,
        generated_at=datetime.now(UTC),
    )
    manifest_tmp.write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    return manifest
