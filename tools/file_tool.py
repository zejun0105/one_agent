"""File read/write tool."""

import json
import uuid
from pathlib import Path
from typing import Any, Optional
from .base import Tool, ToolResult


class FileReadTool(Tool):
    """Tool for reading files."""

    def __init__(
        self,
        name: str = "file_read",
        description: str = "Read the contents of a file",
        allowed_dirs: Optional[list] = None
    ):
        """Initialize file read tool.

        Args:
            name: Tool name
            description: Tool description
            allowed_dirs: List of allowed directories (None means all)
        """
        super().__init__(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (default: utf-8)"
                    }
                },
                "required": ["path"]
            }
        )
        self.allowed_dirs = allowed_dirs or []

    def _is_allowed(self, path: str) -> bool:
        """Check if path is in allowed directories."""
        if not self.allowed_dirs:
            return True
        path_obj = Path(path).resolve()
        for allowed in self.allowed_dirs:
            allowed_path = Path(allowed).resolve()
            try:
                path_obj.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        return False

    def execute(self, path: str = None, encoding: str = "utf-8", **kwargs) -> ToolResult:
        """Read file contents.

        Args:
            path: Path to file
            encoding: File encoding (auto-detect if utf-8 fails)

        Returns:
            ToolResult with file contents
        """
        # Check required parameter
        if path is None:
            return ToolResult(
                success=False,
                content="",
                error="Missing required parameter: 'path'",
                tool_call_id=f"read_{uuid.uuid4().hex[:8]}"
            )

        tool_id = f"read_{uuid.uuid4().hex[:8]}"

        if not self._is_allowed(path):
            return ToolResult(
                success=False,
                content="",
                error=f"Access denied: path '{path}' is not in allowed directories",
                tool_call_id=tool_id
            )

        try:
            file_path = Path(path)
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"File not found: {path}",
                    tool_call_id=tool_id
                )

            if file_path.is_dir():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Path is a directory, not a file: {path}",
                    tool_call_id=tool_id
                )

            # Try UTF-8 first, then fallback to common encodings
            encodings_to_try = ["utf-8", "gbk", "gb2312", "gb18030", "latin1"]

            content = None
            last_error = None
            for enc in encodings_to_try:
                try:
                    content = file_path.read_text(encoding=enc)
                    break
                except UnicodeDecodeError as e:
                    last_error = e
                    continue

            if content is None:
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Failed to decode file with any encoding: {last_error}",
                    tool_call_id=tool_id
                )

            return ToolResult(
                success=True,
                content=content,
                tool_call_id=tool_id
            )

        except PermissionError:
            return ToolResult(
                success=False,
                content="",
                error=f"Permission denied: {path}",
                tool_call_id=tool_id
            )

        except UnicodeDecodeError as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Failed to decode file with encoding '{encoding}': {str(e)}",
                tool_call_id=tool_id
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Failed to read file: {str(e)}",
                tool_call_id=tool_id
            )


class FileWriteTool(Tool):
    """Tool for writing files."""

    def __init__(
        self,
        name: str = "file_write",
        description: str = "Write content to a file",
        allowed_dirs: Optional[list] = None
    ):
        """Initialize file write tool.

        Args:
            name: Tool name
            description: Tool description
            allowed_dirs: List of allowed directories (None means all)
        """
        super().__init__(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (default: utf-8)"
                    },
                    "mode": {
                        "type": "string",
                        "description": "Write mode: 'w' (overwrite) or 'a' (append)"
                    }
                },
                "required": ["path", "content"]
            }
        )
        self.allowed_dirs = allowed_dirs or []

    def _is_allowed(self, path: str) -> bool:
        """Check if path is in allowed directories."""
        if not self.allowed_dirs:
            return True
        path_obj = Path(path).resolve()
        for allowed in self.allowed_dirs:
            allowed_path = Path(allowed).resolve()
            try:
                path_obj.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        return False

    def execute(
        self,
        path: str = None,
        content: str = None,
        encoding: str = "utf-8",
        mode: str = "w",
        **kwargs
    ) -> ToolResult:
        """Write content to file.

        Args:
            path: Path to file
            content: Content to write
            encoding: File encoding
            mode: Write mode ('w' or 'a')

        Returns:
            ToolResult with result message
        """
        # Check required parameters
        if path is None:
            return ToolResult(
                success=False,
                content="",
                error="Missing required parameter: 'path'",
                tool_call_id=f"write_{uuid.uuid4().hex[:8]}"
            )
        if content is None:
            return ToolResult(
                success=False,
                content="",
                error="Missing required parameter: 'content'",
                tool_call_id=f"write_{uuid.uuid4().hex[:8]}"
            )

        tool_id = f"write_{uuid.uuid4().hex[:8]}"

        if not self._is_allowed(path):
            return ToolResult(
                success=False,
                content="",
                error=f"Access denied: path '{path}' is not in allowed directories",
                tool_call_id=tool_id
            )

        if mode not in ('w', 'a'):
            return ToolResult(
                success=False,
                content="",
                error=f"Invalid mode: '{mode}'. Use 'w' (overwrite) or 'a' (append)",
                tool_call_id=tool_id
            )

        try:
            file_path = Path(path)

            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if mode == 'a':
                file_path.write_text(content, encoding=encoding, append=True)
            else:
                file_path.write_text(content, encoding=encoding)

            action = "appended to" if mode == 'a' else "written to"
            return ToolResult(
                success=True,
                content=f"Successfully {action}: {path}",
                tool_call_id=tool_id
            )

        except PermissionError:
            return ToolResult(
                success=False,
                content="",
                error=f"Permission denied: {path}",
                tool_call_id=tool_id
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Failed to write file: {str(e)}",
                tool_call_id=tool_id
            )
