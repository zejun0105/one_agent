"""Compatible provider for GLM-4, Kimi, and other compatible APIs."""

import json
import re
from typing import Optional, List, Any, Generator
from .base import BaseLLMProvider, LLMResponse, ToolCall, StreamChunk


class CompatibleProvider(BaseLLMProvider):
    """Provider for compatible APIs like GLM-4 (智谱 AI) and Kimi (月之暗面)."""

    # Models that support native tool calling
    NATIVE_TOOL_MODELS = [
        "glm-4",
        "glm-4-plus",
        "glm-4v",
        "moonshot-v1",
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k",
    ]

    def __init__(
        self,
        api_key: str,
        model: str = "glm-4-plus",
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """Initialize Compatible provider.

        Args:
            api_key: API key
            model: Model name to use
            base_url: Base URL for the API
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        self.api_key = api_key
        self._model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.supports_native_tools = self._check_native_support()

        # Import here to avoid dependency if not used
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def _check_native_support(self) -> bool:
        """Check if model supports native tool calling."""
        model_lower = self._model.lower()
        return any(m in model_lower for m in self.NATIVE_TOOL_MODELS)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        # Infer provider from model name
        if "glm" in self._model.lower():
            return "glm"
        elif "moonshot" in self._model.lower() or "kimi" in self._model.lower():
            return "kimi"
        else:
            return "compatible"

    def format_tools(self, tools: List[dict]) -> List[dict]:
        """Format tools for OpenAI-compatible API."""
        formatted = []
        for tool in tools:
            # Ensure parameters have correct schema
            params = tool.get("parameters", {})
            if isinstance(params, dict) and "properties" not in params:
                # Wrap in proper JSON Schema format if needed
                params = {
                    "type": "object",
                    "properties": params.get("properties", {}),
                    "required": params.get("required", [])
                }

            tool_def = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": params
                }
            }
            formatted.append(tool_def)
        return formatted

    def parse_response(self, response) -> LLMResponse:
        """Parse API response."""
        if hasattr(response, 'choices') and response.choices:
            # OpenAI-compatible response format
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

            # Try to parse text-based tool calls if none found
            if not tool_calls and content:
                parsed = self._parse_text_tool_calls(content)
                if parsed:
                    tool_calls = parsed
                    content = self._remove_tool_calls_from_content(content)

            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                }

            return LLMResponse(content=content, tool_calls=tool_calls, usage=usage)

        return LLMResponse(content=str(response))

    def parse_stream_chunk(self, chunk) -> StreamChunk:
        """Parse a streaming chunk."""
        delta = ""
        is_final = False
        tool_calls = None

        if hasattr(chunk, 'choices') and chunk.choices:
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

    def _parse_text_tool_calls(self, content: str) -> Optional[List[ToolCall]]:
        """Parse tool calls from text content (for models without native support)."""
        pattern = r'```tool_call\s*\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            return None

        tool_calls = []
        for i, match in enumerate(matches):
            try:
                call_data = json.loads(match)
                tool_calls.append(ToolCall(
                    id=f"call_{i}",
                    name=call_data["tool"],
                    arguments=call_data.get("parameters", {})
                ))
            except (json.JSONDecodeError, KeyError):
                continue

        return tool_calls if tool_calls else None

    def _remove_tool_calls_from_content(self, content: str) -> str:
        """Remove tool call blocks from content."""
        pattern = r'```tool_call\s*\n.*?\n```'
        return re.sub(pattern, '', content, flags=re.DOTALL).strip()

    def _format_tools_as_text(self, tools: List[dict]) -> str:
        """Format tools as text for system prompt."""
        text = "# Available Tools\n\n"
        for tool in tools:
            text += f"## {tool['name']}\n"
            text += f"{tool.get('description', 'No description')}\n\n"
            text += "Parameters:\n```json\n"
            text += json.dumps(tool.get('parameters', {}), indent=2, ensure_ascii=False)
            text += "\n```\n\n"

        text += """
When using a tool, format your response as:
```tool_call
{
  "tool": "tool_name",
  "parameters": {parameters}
}
```
"""
        return text

    def _format_tool_result(self, message: dict) -> dict:
        """Format tool result message for GLM API."""
        # GLM API requires 'type' field for tool messages
        if self.provider_name == "glm" and message.get("role") == "tool":
            return {
                "role": "tool",
                "content": message.get("content", ""),
                "tool_call_id": message.get("tool_call_id", ""),
                "type": "tool"  # GLM requires this field
            }
        return message

    def _format_messages(self, messages: List[dict]) -> List[dict]:
        """Format messages for the API provider."""
        if self.provider_name == "glm":
            return [self._format_tool_result(msg) for msg in messages]
        return messages

    def chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat request to compatible API."""
        if not self.supports_native_tools and tools:
            # Use text-based tool calling (fallback mode)
            return self._chat_with_text_tools(messages, tools, **kwargs)
        else:
            # Use native tool calling
            return self._chat_native(messages, tools, **kwargs)

    def stream(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs
    ) -> Generator[StreamChunk, None, None]:
        """Stream chat request to compatible API."""
        # Note: Streaming with tools is limited for compatible providers
        # Text-based tool calling doesn't support streaming well

        if not self.supports_native_tools and tools:
            # Fallback to non-streaming with text-based tools
            response = self._chat_with_text_tools(messages, tools, **kwargs)
            # Simulate streaming with single chunk
            yield StreamChunk(
                content=response.content or "",
                delta=response.content or "",
                is_final=True,
                tool_calls=response.tool_calls
            )
            return

        # Native streaming
        # Format messages for GLM API
        formatted_messages = self._format_messages(messages)
        params = {
            "model": self._model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **kwargs
        }

        if tools:
            params["tools"] = self.format_tools(tools)
            params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**params)

        full_content = ""
        for chunk in response:
            parsed = self.parse_stream_chunk(chunk)
            full_content += parsed.delta

            yield StreamChunk(
                content=full_content,
                delta=parsed.delta,
                is_final=parsed.is_final,
                tool_calls=parsed.tool_calls
            )

            if parsed.is_final:
                break

    def _chat_native(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """Chat with native tool support."""
        # Format messages for GLM API
        formatted_messages = self._format_messages(messages)

        params = {
            "model": self._model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }

        if tools:
            # Format tools based on provider
            formatted_tools = self.format_tools(tools)

            if self.provider_name == "glm":
                # GLM API format
                params["tools"] = formatted_tools
                params["tool_choice"] = "auto"
            else:
                # Standard OpenAI-compatible format
                params["tools"] = formatted_tools
                params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**params)
        return self.parse_response(response)

    def _chat_with_text_tools(
        self,
        messages: List[dict],
        tools: List[dict],
        **kwargs
    ) -> LLMResponse:
        """Chat with text-based tool calling (for models without native support)."""
        # Add tool descriptions to system message
        tool_descriptions = self._format_tools_as_text(tools)

        enhanced_messages = []
        for msg in messages:
            if msg["role"] == "system":
                enhanced_messages.append({
                    "role": "system",
                    "content": msg["content"] + "\n\n" + tool_descriptions
                })
            else:
                enhanced_messages.append(msg)

        # Add system message if not present
        if not any(m["role"] == "system" for m in enhanced_messages):
            enhanced_messages.insert(0, {
                "role": "system",
                "content": tool_descriptions
            })

        # Make API call without tools
        params = {
            "model": self._model,
            "messages": enhanced_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }

        response = self.client.chat.completions.create(**params)

        # Parse response and extract tool calls from text
        return self.parse_response(response)
