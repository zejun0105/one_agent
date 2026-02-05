"""Anthropic (Claude) provider implementation."""

import json
from typing import Optional, List, Any
from .base import BaseLLMProvider, LLMResponse, ToolCall


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

    def chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat request to Anthropic."""
        # Extract system message (Anthropic requires it separately)
        system_message = None
        api_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                api_messages.append(msg)

        # Build request parameters
        params = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **kwargs
        }

        # Add system message if present
        if system_message:
            params["system"] = system_message

        # Add tools if present
        if tools:
            params["tools"] = self.format_tools(tools)
            params["tool_choice"] = {"type": "auto"}

        # Make API call
        response = self.client.messages.create(**params)

        return self.parse_response(response)
