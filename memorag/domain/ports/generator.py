from abc import ABC, abstractmethod
from collections.abc import Generator as PyGenerator


class Generator(ABC):
    @abstractmethod
    def generate(self, context: str, query: str) -> PyGenerator[str, None, None]:
        """Generates a natural language response based on the context and the user query."""
        pass
