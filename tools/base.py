"""Base tool classes for One-Agent."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """Result from executing a tool."""
    success: bool
    content: str
    tool_call_id: str = "unknown"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
            "error": self.error
        }


class Tool(ABC):
    """Abstract base class for tools."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
    ):
        """Initialize a tool.

        Args:
            name: Tool name (should be lowercase with underscores)
            description: Description of what the tool does
            parameters: JSON Schema for parameters
        """
        self.name = name
        self.description = description
        self.parameters = parameters

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool.

        Args:
            **kwargs: Tool parameters

        Returns:
            ToolResult with success status and content
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to tool definition dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

    def __repr__(self) -> str:
        return f"Tool(name={self.name}, description={self.description[:50]}...)"
