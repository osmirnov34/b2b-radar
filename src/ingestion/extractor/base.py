from typing import Protocol

from src.domain.document import Document
from src.domain.source import Source


class Extractor(Protocol):
    """Common interface for extractors that turn a search query into Source and Document objects."""

    def extract_sources(self, query: str, limit: int = 50) -> list[Source]:
        """Search for items matching a query and map them to Source objects.

        Args:
            query (str): The search query.
            limit (int, optional): The maximum number of items to extract. Defaults to 50.

        Returns:
            list[Source]: Extracted sources.

        """
        ...

    def extract_documents(self, source: Source, limit: int = 100) -> list[Document]:
        """Fetch items (e.g. comments) attached to a persisted Source and map them to Document objects.

        Args:
            source (Source): A persisted source (must have an id).
            limit (int, optional): The maximum number of documents to extract. Defaults to 100.

        Returns:
            list[Document]: Extracted documents.

        """
        ...
