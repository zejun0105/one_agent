"""Base LLM Provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Any, Generator


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[dict] = None

    def __post_init__(self):
        if self.tool_calls and isinstance(self.tool_calls[0], dict):
            self.tool_calls = [ToolCall(**tc) for tc in self.tool_calls]


@dataclass
class StreamChunk:
    """A chunk from a streaming response."""
    content: str = ""
    delta: str = ""
    is_final: bool = False
    tool_calls: Optional[List[ToolCall]] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """Send a chat request to the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: Optional list of tool definitions
            **kwargs: Additional provider-specific arguments

        Returns:
            LLMResponse with content and/or tool calls
        """
        pass

    @abstractmethod
    def stream(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs
    ) -> Generator[StreamChunk, None, None]:
        """Stream a chat request to the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: Optional list of tool definitions
            **kwargs: Additional provider-specific arguments

        Yields:
            StreamChunk with content deltas
        """
        pass

    @abstractmethod
    def format_tools(self, tools: List[dict]) -> Any:
        """Format tools for this provider's API.

        Args:
            tools: List of tool definition dictionaries

        Returns:
            Provider-specific tool format
        """
        pass

    @abstractmethod
    def parse_response(self, response: Any) -> LLMResponse:
        """Parse an API response into LLMResponse format.

        Args:
            response: Raw API response

        Returns:
            Parsed LLMResponse
        """
        pass

    @abstractmethod
    def parse_stream_chunk(self, chunk: Any) -> StreamChunk:
        """Parse a streaming chunk.

        Args:
            chunk: Raw streaming chunk

        Returns:
            Parsed StreamChunk
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
