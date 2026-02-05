"""Anthropic (Claude) provider implementation."""

import json
from typing import Optional, List, Any, Generator
from .base import BaseLLMProvider, LLMResponse, ToolCall, StreamChunk


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic's Claude models."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Model name to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        self.api_key = api_key
        self._model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Import here to avoid dependency if not used
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def format_tools(self, tools: List[dict]) -> List[dict]:
        """Format tools for Anthropic's API."""
        formatted = []
        for tool in tools:
            formatted.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool.get("parameters", {})
            })
        return formatted

    def parse_response(self, response) -> LLMResponse:
        """Parse Anthropic API response."""
        content = None
        tool_calls = None
        usage = None

        # Get the message content
        if response.content:
            for block in response.content:
                if block.type == "text":
                    content = block.text
                elif block.type == "tool_use":
                    tool_calls = [
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input
                        )
                    ]

        # Get usage info
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage
        )

    def parse_stream_chunk(self, chunk) -> StreamChunk:
        """Parse an Anthropic streaming chunk."""
        delta = ""
        is_final = False
        tool_calls = None

        if chunk.type == "content_block_delta":
            if chunk.delta.type == "text_delta":
                delta = chunk.delta.text
            elif chunk.delta.type == "tool_use_delta":
                # Tool use in streaming - capture the full input
                if hasattr(chunk.delta, 'input') and chunk.delta.input:
                    delta = json.dumps(chunk.delta.input, ensure_ascii=False)

        elif chunk.type == "message_delta":
            if hasattr(chunk, 'stop_reason'):
                is_final = True

        return StreamChunk(
            delta=delta,
            is_final=is_final,
            tool_calls=tool_calls
        )

    def _prepare_params(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
    ) -> tuple:
        """Prepare request parameters.

        Returns:
            Tuple of (system_message, api_messages, params)
        """
        system_message = None
        api_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                api_messages.append(msg)

        params = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        # Add system message if present
        if system_message:
            params["system"] = system_message

        # Add tools if present
        if tools:
            params["tools"] = self.format_tools(tools)
            params["tool_choice"] = {"type": "auto"}

        return system_message, api_messages, params

    def chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat request to Anthropic."""
        _, _, params = self._prepare_params(messages, tools)
        params.update(kwargs)

        # Make API call
        response = self.client.messages.create(**params)

        return self.parse_response(response)

    def stream(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs
    ) -> Generator[StreamChunk, None, None]:
        """Stream chat request to Anthropic."""
        _, _, params = self._prepare_params(messages, tools)
        params.update(kwargs)

        # Stream the response
        with self.client.messages.stream(**params) as stream:
            full_content = ""

            for chunk in stream:
                parsed = self.parse_stream_chunk(chunk)
                full_content += parsed.delta

                # Check for completion
                if parsed.is_final or chunk.type == "message_stop":
                    yield StreamChunk(
                        content=full_content,
                        delta=parsed.delta,
                        is_final=True,
                        tool_calls=parsed.tool_calls
                    )
                    break

                # Yield partial chunks
                if parsed.delta:
                    yield StreamChunk(
                        content=full_content,
                        delta=parsed.delta,
                        is_final=False
                    )
