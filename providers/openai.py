"""OpenAI (GPT-4) provider implementation."""

import json
from typing import Optional, List, Any, Generator
from .base import BaseLLMProvider, LLMResponse, ToolCall, StreamChunk


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI's GPT-4 models."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo",
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name to use
            base_url: Optional base URL for API (for compatible providers)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        self.api_key = api_key
        self._model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Import here to avoid dependency if not used
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "openai"

    def format_tools(self, tools: List[dict]) -> List[dict]:
        """Format tools for OpenAI's Function Calling API."""
        formatted = []
        for tool in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool.get("parameters", {})
                }
            })
        return formatted

    def parse_response(self, response) -> LLMResponse:
        """Parse OpenAI API response."""
        message = response.choices[0].message

        content = message.content
        tool_calls = None

        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                )
                for tc in message.tool_calls
            ]

        # Get usage info
        usage = None
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage
        )

    def parse_stream_chunk(self, chunk) -> StreamChunk:
        """Parse an OpenAI streaming chunk."""
        delta = ""
        is_final = False
        tool_calls = None

        if chunk.choices and len(chunk.choices) > 0:
            choice = chunk.choices[0]

            if choice.delta:
                if choice.delta.content:
                    delta = choice.delta.content

                # Check for tool calls
                if choice.delta.tool_calls:
                    tool_calls = []
                    for tc in choice.delta.tool_calls:
                        if tc.function:
                            arguments = tc.function.arguments or ""
                            tool_calls.append(ToolCall(
                                id=tc.id or f"call_{len(tool_calls)}",
                                name=tc.function.name,
                                arguments=json.loads(arguments) if arguments else {}
                            ))

            # Check for completion
            if choice.finish_reason:
                is_final = True

        return StreamChunk(
            delta=delta,
            is_final=is_final,
            tool_calls=tool_calls
        )

    def chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat request to OpenAI."""
        # Build request parameters
        params = {
            "model": self._model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }

        # Add tools if present
        if tools:
            params["tools"] = self.format_tools(tools)
            params["tool_choice"] = "auto"

        # Make API call
        response = self.client.chat.completions.create(**params)

        return self.parse_response(response)

    def stream(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs
    ) -> Generator[StreamChunk, None, None]:
        """Stream chat request to OpenAI."""
        # Build request parameters
        params = {
            "model": self._model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **kwargs
        }

        # Add tools if present
        if tools:
            params["tools"] = self.format_tools(tools)
            params["tool_choice"] = "auto"

        # Make streaming API call
        response = self.client.chat.completions.create(**params)

        full_content = ""
        for chunk in response:
            parsed = self.parse_stream_chunk(chunk)
            full_content += parsed.delta

            # Yield partial chunks
            yield StreamChunk(
                content=full_content,
                delta=parsed.delta,
                is_final=parsed.is_final,
                tool_calls=parsed.tool_calls
            )

            # Stop after final chunk
            if parsed.is_final:
                break
