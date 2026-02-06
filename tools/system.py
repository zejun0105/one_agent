"""System command execution tool."""

import json
import uuid
import subprocess
from typing import Any, Optional, List
from .base import Tool, ToolResult


class SystemCommandTool(Tool):
    """Tool for executing system commands."""

    def __init__(
        self,
        name: str = "system",
        description: str = "Execute a system command",
        allowed_commands: Optional[List[str]] = None,
        timeout: int = 60
    ):
        """Initialize system command tool.

        Args:
            name: Tool name
            description: Tool description
            allowed_commands: List of allowed commands (None means all)
            timeout: Command timeout in seconds
        """
        super().__init__(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute (use spaces to separate arguments)"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60)"
                    },
                    "shell": {
                        "type": "boolean",
                        "description": "Use shell execution (default: false)"
                    }
                },
                "required": ["command"]
            }
        )
        self.allowed_commands = allowed_commands or []
        self.timeout = timeout

    def _is_allowed(self, command: str) -> bool:
        """Check if command is in allowed list."""
        if not self.allowed_commands:
            return True

        # Get the base command
        base_cmd = command.strip().split()[0] if command.strip() else ""

        for allowed in self.allowed_commands:
            if base_cmd == allowed:
                return True
            # Also check if it's a path to an allowed command
            if "/" in base_cmd or "\\" in base_cmd:
                base_name = base_cmd.split("/")[-1].split("\\")[-1]
                if base_name == allowed:
                    return True

        return False

    def execute(
        self,
        command: str = None,
        timeout: int = None,
        shell: bool = False,
        **kwargs
    ) -> ToolResult:
        """Execute a system command.

        Args:
            command: Command to execute
            timeout: Optional timeout override
            shell: Whether to use shell execution

        Returns:
            ToolResult with command output
        """
        tool_id = f"system_{uuid.uuid4().hex[:8]}"

        # Check required parameter
        if command is None:
            return ToolResult(
                success=False,
                content="",
                error="Missing required parameter: 'command'",
                tool_call_id=tool_id
            )

        # Security check
        if not self._is_allowed(command):
            return ToolResult(
                success=False,
                content="",
                error=f"Command not allowed: {command.split()[0] if command else 'empty'}",
                tool_call_id=tool_id
            )

        try:
            # Parse command and arguments
            import shlex
            args = shlex.split(command) if not shell else None

            # Execute command
            exec_timeout = timeout or self.timeout
            result = subprocess.run(
                args,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=exec_timeout
            )

            output = ""
            if result.stdout:
                output += f"[stdout]\n{result.stdout}"
            if result.stderr:
                output += f"[stderr]\n{result.stderr}"

            if not output:
                output = "Command executed successfully (no output)"

            # Check if command succeeded
            success = result.returncode == 0

            return ToolResult(
                success=success,
                content=output.strip(),
                tool_call_id=tool_id,
                error=f"Exit code: {result.returncode}" if result.returncode != 0 else None
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                content="",
                error=f"Command timed out after {exec_timeout} seconds",
                tool_call_id=tool_id
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Command failed: {str(e)}",
                tool_call_id=tool_id
            )
