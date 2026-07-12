from typing import Any, cast

from googleapiclient.discovery import build


class YoutubeClient:
    """Client for YouTube platform using official Data API v3."""

    def __init__(self, api_keys: list[str]) -> None:
        self.api_keys = api_keys
        self.current_key_index = 0
        self.youtube = build("youtube", "v3", developerKey=self.api_keys[self.current_key_index])

    def search_videos(self, query: str, max_results: int = 50) -> list[dict[str, Any]]:
        """Search for videos on YouTube based on a given query.

        Args:
            query (str): The search query.
            max_results (int, optional): The maximum number of results to return. Defaults to 50.

        Returns:
            list[dict[str, Any]]: A list of video search results.

        """
        request = self.youtube.search().list(q=query, part="snippet", maxResults=max_results, type="video")
        response: dict[str, Any] = request.execute()
        return cast("list[dict[str, Any]]", response.get("items", []))
