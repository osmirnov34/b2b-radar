from typing import Protocol

from src.domain.source import Source


class Extractor(Protocol):
    """Common interface for extractors that turn a search query into Source objects."""

    def extract_sources(self, query: str, limit: int = 50) -> list[Source]:
        """Search for items matching a query and map them to Source objects.

        Args:
            query (str): The search query.
            limit (int, optional): The maximum number of items to extract. Defaults to 50.

        Returns:
            list[Source]: Extracted sources.

        """
        ...
