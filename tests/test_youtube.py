from datetime import UTC

import pytest

from src.infrastructure.api.youtube import YoutubeClient
from src.infrastructure.extractor.youtube import YoutubeExtractor


class _FlakyRequest:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self) -> dict[str, object]:
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError
        return {"items": [{"id": "ok"}]}


def test_youtube_request_retries_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _FlakyRequest()
    monkeypatch.setattr("src.infrastructure.api.youtube.time.sleep", lambda _seconds: None)

    assert YoutubeClient._execute_with_retry(request) == {"items": [{"id": "ok"}]}  # type: ignore[arg-type]
    assert request.calls == 2


def test_video_resource_is_mapped_to_source() -> None:
    source = YoutubeExtractor._to_source(
        {
            "id": "video-1",
            "snippet": {
                "title": "CRM",
                "description": "Description",
                "publishedAt": "2026-01-01T10:00:00Z",
                "channelId": "channel-1",
                "channelTitle": "Channel",
            },
            "statistics": {"viewCount": "42", "likeCount": "7"},
            "contentDetails": {"duration": "PT5M", "caption": "true"},
        },
    )

    assert source.url == "https://www.youtube.com/watch?v=video-1"
    assert source.metadata["statistics"]["view_count"] == 42
    assert source.metadata["content_details"]["has_captions"] is True


def test_comment_thread_is_mapped_with_replies() -> None:
    document = YoutubeExtractor._to_document(
        10,
        {
            "snippet": {
                "topLevelComment": {
                    "id": "comment-1",
                    "snippet": {
                        "textDisplay": "Comment",
                        "publishedAt": "2026-01-01T10:00:00Z",
                        "authorDisplayName": "Author",
                        "likeCount": 3,
                    },
                },
                "totalReplyCount": 2,
            },
            "replies": {
                "comments": [
                    {
                        "snippet": {
                            "textDisplay": "Reply",
                            "publishedAt": "2026-01-01T11:00:00Z",
                            "authorDisplayName": "Responder",
                            "likeCount": 1,
                        },
                    },
                ],
            },
        },
    )

    assert document.source_id == 10
    assert document.created_at.tzinfo == UTC
    assert document.metadata["total_reply_count"] == 2
    assert document.metadata["replies"][0]["text"] == "Reply"
