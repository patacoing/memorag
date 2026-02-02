from unittest.mock import Mock, patch

from memorag.infrastructure.adapters.litellm_generator import LiteLLMGenerator, LiteLLMRole


class TestLiteLLMGenerator:
    def test_system_message_structure(self):
        generator = LiteLLMGenerator("model-name")
        msg = generator.system_message

        assert msg["role"] == LiteLLMRole.SYSTEM
        assert "helpful assistant" in msg["content"]

    def test_build_user_prompt_format(self):
        generator = LiteLLMGenerator("model-name")
        context = "some context"
        query = "user question"

        prompt = generator._build_user_prompt(context, query)

        assert "Context:" in prompt
        assert context in prompt
        assert "Question:" in prompt
        assert query in prompt
        assert "Answer:" in prompt

    @patch("memorag.infrastructure.adapters.litellm_generator.litellm")
    def test_generate_calls_litellm_correctly(self, mock_litellm):
        # Setup
        generator = LiteLLMGenerator("test-model")
        context = "ctx"
        query = "q"

        # Mock streaming response
        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta = {"content": "response token"}

        mock_litellm.completion.return_value = [mock_chunk]

        # Execute
        list(generator.generate(context, query))

        # Verify
        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args.kwargs

        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["stream"] is True

        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == LiteLLMRole.SYSTEM
        assert messages[1]["role"] == LiteLLMRole.USER
