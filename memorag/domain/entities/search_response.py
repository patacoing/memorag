from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from .vector import Vector


class SearchResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    answer: Iterable[str]
    sources: list[Vector]
