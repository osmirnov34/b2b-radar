from datetime import datetime

from pydantic import BaseModel


class ApiKey(BaseModel):
    """A managed YouTube Data API key."""

    id: int | None = None
    name: str | None = None
    key: str
    is_active: bool = True
    created_at: datetime | None = None

    @property
    def masked_key(self) -> str:
        return f"{self.key[:4]}{'•' * 12}{self.key[-4:]}" if len(self.key) > 8 else "••••••••"  # noqa: PLR2004
