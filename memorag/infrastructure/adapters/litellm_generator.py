from collections.abc import Generator as PyGenerator
from enum import Enum
from typing import TypedDict

import litellm

from memorag.domain.ports import Generator


class LiteLLMRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class LiteLLMMessage(TypedDict):
    role: LiteLLMRole
    content: str


class LiteLLMGenerator(Generator):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @property
    def system_prompt(self) -> str:
        return """You are a helpful assistant. Use the provided context to 
answer the user's question. If the answer is not in the context, say so gracefully."""

    @property
    def system_message(self) -> LiteLLMMessage:
        return self._build_message(LiteLLMRole.SYSTEM, self.system_prompt)

    def _build_message(self, role: LiteLLMRole, content: str) -> LiteLLMMessage:
        return LiteLLMMessage(role=role, content=content)

    def _build_user_prompt(self, context: str, query: str) -> str:
        return f"""
Context:
{context}

Question: 
{query}

Answer:
"""

    def generate(self, context: str, query: str) -> PyGenerator[str, None, None]:
        user_content = self._build_user_prompt(context, query)
        messages = [self.system_message, self._build_message(LiteLLMRole.USER, user_content)]

        try:
            response = litellm.completion(model=self.model_name, messages=messages, stream=True)
        except litellm.exceptions.AuthenticationError as e:
            raise RuntimeError("LiteLLM authentication failed. Please check your API key.") from e

        for chunk in response:
            content = chunk.choices[0].delta.get("content", "")  # pyright: ignore
            if content:
                yield content
