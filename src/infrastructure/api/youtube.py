from typing import Any, cast

from googleapiclient.discovery import build


class YoutubeClient:
    """Client for YouTube platform using official Data API v3."""

    def __init__(self, api_keys: list[str]) -> None:
        self.api_keys = api_keys
        self.current_key_index = 0
        self.youtube = build("youtube", "v3", developerKey=self.api_keys[self.current_key_index])

    def search_videos(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """Search for videos on YouTube with pagination support.

        Args:
            query (str): The search query.
            limit (int, optional): The maximum number of results to return. Defaults to 50.

        Returns:
            list[dict[str, Any]]: A list of video search results.

        """
        results: list[dict[str, Any]] = []
        next_page_token: str | None = None

        while len(results) < limit:
            batch_size = min(50, limit - len(results))

            request = self.youtube.search().list(
                q=query,
                part="snippet",
                maxResults=batch_size,
                type="video",
                pageToken=next_page_token,
                order="relevance",
            )
            response = cast("dict[str, Any]", request.execute())

            items = cast("list[dict[str, Any]]", response.get("items", []))
            if not items:
                break

            results.extend(items)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        return results
