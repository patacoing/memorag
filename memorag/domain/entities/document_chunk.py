from uuid import UUID

from pydantic import BaseModel


class DocumentChunk(BaseModel):
    id: UUID
    content: bytes
    metadata: dict[str, str] | None = None
    next_chunk: "DocumentChunk | None" = None
