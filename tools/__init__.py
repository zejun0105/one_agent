"""One-Agent Tools Module

Provides tool implementations for agent capabilities:
- Web search
- Calculator
- Custom tools
"""

from .base import Tool, ToolResult
from .web_search import WebSearchTool
from .calculator import CalculatorTool

__all__ = [
    "Tool",
    "ToolResult",
    "WebSearchTool",
    "CalculatorTool",
]
