"""Python code execution tool."""

import json
import uuid
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Any
from .base import Tool, ToolResult


class PythonCodeTool(Tool):
    """Tool for executing Python code safely."""

    def __init__(
        self,
        name: str = "python_code",
        description: str = "Execute Python code safely",
        timeout: int = 30
    ):
        """Initialize Python code execution tool.

        Args:
            name: Tool name
            description: Tool description
            timeout: Execution timeout in seconds
        """
        super().__init__(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30)"
                    }
                },
                "required": ["code"]
            }
        )
        self.timeout = timeout

    def execute(self, code: str = None, timeout: int = None, **kwargs) -> ToolResult:
        """Execute Python code.

        Args:
            code: Python code to execute
            timeout: Optional timeout override

        Returns:
            ToolResult with execution output
        """
        # Check required parameter
        if code is None:
            return ToolResult(
                success=False,
                content="",
                error="Missing required parameter: 'code'",
                tool_call_id=f"python_{uuid.uuid4().hex[:8]}"
            )

        exec_timeout = timeout or self.timeout
        tool_id = f"python_{uuid.uuid4().hex[:8]}"

        try:
            # Create a sandboxed environment
            sandbox_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'sorted': sorted,
                    'reversed': reversed,
                    'sum': sum,
                    'min': min,
                    'max': max,
                    'abs': abs,
                    'round': round,
                    'pow': pow,
                    'divmod': divmod,
                    'isinstance': isinstance,
                    'type': type,
                    'hasattr': hasattr,
                    'getattr': getattr,
                    'setattr': setattr,
                    'delattr': delattr,
                    'chr': chr,
                    'ord': ord,
                    'bin': bin,
                    'oct': oct,
                    'hex': hex,
                    'format': format,
                    'slice': slice,
                    'property': property,
                    'classmethod': classmethod,
                    'staticmethod': staticmethod,
                    'super': super,
                    'object': object,
                    'BaseException': BaseException,
                    'Exception': Exception,
                    'ValueError': ValueError,
                    'TypeError': TypeError,
                    'KeyError': KeyError,
                    'IndexError': IndexError,
                    'AttributeError': AttributeError,
                    'NameError': NameError,
                    'RuntimeError': RuntimeError,
                }
            }

            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            # Execute the code
            result = None
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                try:
                    # Execute code and get the result if there's a final expression
                    exec(code, sandbox_globals)
                except Exception as e:
                    raise e

            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            output = ""
            if stdout_output:
                output += f"[stdout]\n{stdout_output}"
            if stderr_output:
                output += f"[stderr]\n{stderr_output}"

            if not output:
                output = "Code executed successfully (no output)"

            return ToolResult(
                success=True,
                content=output.strip(),
                tool_call_id=tool_id
            )

        except SyntaxError as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Syntax error: {str(e)}",
                tool_call_id=tool_id
            )

        except Exception as e:
            error_type = type(e).__name__
            return ToolResult(
                success=False,
                content="",
                error=f"{error_type}: {str(e)}",
                tool_call_id=tool_id
            )
