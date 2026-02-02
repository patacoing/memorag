from abc import ABC, abstractmethod
from collections.abc import Generator


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> Generator[str, None, None]:
        pass
