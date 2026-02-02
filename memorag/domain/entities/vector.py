from datetime import datetime
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict


class Vector(BaseModel):
    id: UUID
    vector: np.ndarray
    inserted_at: datetime
    content: bytes
    metadata: dict | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def document_name(self) -> str:
        return (self.metadata or {}).get("document_name", "unknown")

    @property
    def document_id(self) -> str:
        return (self.metadata or {}).get("document_id", "unknown id")
