from src.web.app import create_app
from src.web.routers.analysis import _parse_clusters


def test_cluster_parser_keeps_only_valid_cluster_records() -> None:
    raw = b'not-json\n{"metadata": true}\n{"topic_id": 1, "comments": []}\n'

    assert _parse_clusters(raw) == [{"topic_id": 1, "comments": []}]


def test_expected_web_routes_are_registered() -> None:
    paths: set[str] = set()
    pending = list(create_app().routes)
    while pending:
        route = pending.pop()
        if path := getattr(route, "path", None):
            paths.add(path)
        pending.extend(getattr(route, "routes", []))
        if original_router := getattr(route, "original_router", None):
            pending.extend(original_router.routes)

    assert {
        "/",
        "/videos",
        "/queries",
        "/comments",
        "/comments/export.jsonl",
        "/analysis",
        "/api-keys",
        "/api/topics",
        "/api/topics/{topic_id}",
        "/api/topics/{topic_id}/assignments",
        "/api/ml/outliers",
        "/api/ml/quality",
    } <= paths
