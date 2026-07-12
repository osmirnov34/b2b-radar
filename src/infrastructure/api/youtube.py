from googleapiclient.discovery import build


class YoutubeClient:
    """Client for YouTube platform using official Data API v3."""

    def __init__(self, api_keys: list[str]) -> None:
        self.api_keys = api_keys
        self.current_key_index = 0
        self.youtube = build("youtube", "v3", developerKey=self.api_keys[self.current_key_index])
