"""Calculator tool implementation."""

import math
import json
import uuid
from typing import Any
from .base import Tool, ToolResult


class CalculatorTool(Tool):
    """Tool for performing mathematical calculations."""

    def __init__(
        self,
        name: str = "calculator",
        description: str = "Perform mathematical calculations"
    ):
        """Initialize calculator tool."""
        super().__init__(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', '10 * 5')"
                    }
                },
                "required": ["expression"]
            }
        )

    def execute(self, expression: str = None, **kwargs) -> ToolResult:
        """Execute calculator operation.

        Args:
            expression: Mathematical expression

        Returns:
            ToolResult with calculation result
        """
        # Check required parameter
        if expression is None:
            return ToolResult(
                success=False,
                content="",
                error="Missing required parameter: 'expression'",
                tool_call_id=f"calc_{uuid.uuid4().hex[:8]}"
            )

        try:
            # Clean the expression
            expression = expression.strip()

            # Define available math functions for eval
            safe_globals = {
                '__builtins__': {},
                'abs': abs,
                'round': round,
                'min': min,
                'max': max,
                'sum': sum,
                'pow': pow,
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'log10': math.log10,
                'exp': math.exp,
                'pi': math.pi,
                'e': math.e,
                'factorial': math.factorial,
                'gcd': math.gcd,
                'lcm': math.lcm,
                'floor': math.floor,
                'ceil': math.ceil,
                'trunc': math.trunc,
            }

            # Evaluate the expression
            result = eval(expression, safe_globals)

            # Format the result
            if isinstance(result, float):
                # Round to reasonable precision
                if abs(result) < 1e-10:
                    result = 0.0
                elif abs(result - round(result)) < 1e-10:
                    result = int(round(result))
                else:
                    result = round(result, 10)

            content = json.dumps({"expression": expression, "result": result}, ensure_ascii=False)

            return ToolResult(
                success=True,
                content=content,
                tool_call_id=f"calc_{uuid.uuid4().hex[:8]}"
            )

        except SyntaxError as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Invalid expression syntax: {str(e)}",
                tool_call_id=f"calc_{uuid.uuid4().hex[:8]}"
            )

        except (NameError, TypeError, ValueError) as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Evaluation error: {str(e)}",
                tool_call_id=f"calc_{uuid.uuid4().hex[:8]}"
            )

        except ZeroDivisionError:
            return ToolResult(
                success=False,
                content="",
                error="Division by zero",
                tool_call_id=f"calc_{uuid.uuid4().hex[:8]}"
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Unexpected error: {str(e)}",
                tool_call_id=f"calc_{uuid.uuid4().hex[:8]}"
            )
