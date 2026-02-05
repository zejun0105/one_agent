# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

One-Agent is a modular Multi-Model Business Agent framework with full SDK integration, multi-provider support, conversation history persistence, streaming response, and comprehensive tool system.

## Tech Stack

- **Language**: Python 3.10+
- **Core SDK**: claude-agent-sdk 0.1.29
- **LLM Providers**: Anthropic, OpenAI, GLM-4, Kimi
- **Dependencies**: pydantic, python-dotenv, requests, colorama

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Interactive mode
PYTHONPATH=. python main.py

# Single query
PYTHONPATH=. python main.py --query "What is 25 * 4?"

# Use specific provider
PYTHONPATH=. python main.py --provider openai --query "Hello"

# Streaming response
PYTHONPATH=. python main.py --stream --query "Tell me a story"

# Verbose output with tool execution details
PYTHONPATH=. python main.py --verbose --query "Search for latest AI news"

# List configured providers
PYTHONPATH=. python main.py --list-providers

# History management
PYTHONPATH=. python main.py --list-sessions              # List all saved sessions
PYTHONPATH=. python main.py --save-history [NAME]       # Save current history
PYTHONPATH=. python main.py --load-history NAME          # Load a session
PYTHONPATH=. python main.py --clear-history [NAME]      # Clear a session
PYTHONPATH=. python main.py --export-history NAME PATH   # Export to file

# Custom env file
PYTHONPATH=. python main.py --env /path/to/.env
```

## Architecture

### Module Structure

```
one-agent/
├── main.py              # Entry point, CLI, agent factory
├── core/                # Core agent logic
│   ├── __init__.py      # Exports: Agent, Config, ConversationHistory, SessionMetadata
│   ├── agent.py         # Main Agent class with tool execution loop, streaming support
│   ├── config.py        # Pydantic models for configuration
│   └── history.py       # Conversation history with persistence
├── providers/           # LLM provider implementations
│   ├── __init__.py      # Exports all providers, StreamChunk, ToolCall
│   ├── base.py          # Abstract base class, LLMResponse, StreamChunk
│   ├── anthropic.py     # Anthropic Claude (native streaming)
│   ├── openai.py        # OpenAI GPT-4 (native streaming)
│   └── compatible.py     # GLM-4, Kimi (native + text fallback)
└── tools/              # Tool implementations
    ├── __init__.py      # Exports all tools
    ├── base.py          # Abstract Tool class, ToolResult
    ├── web_search.py    # DuckDuckGo web search
    ├── calculator.py    # Mathematical expression evaluator
    ├── python_code.py   # Python code execution
    ├── file_tool.py     # File read/write
    ├── system.py        # System command execution
    └── wikipedia.py      # Wikipedia search
```

### Component Interaction

```
CLI (main.py)
    │
    ├─→ Config.load() ──→ .env
    │
    ├─→ create_agent(provider_name)
    │   ├─→ ProviderFactory.create_provider()
    │   │   └─→ AnthropicProvider | OpenAIProvider | CompatibleProvider
    │   │
    │   └─→ create_tools()
    │       └─→ All enabled tools
    │
    └─→ Agent
        ├─→ ConversationHistory (auto-save to ~/.one_agent/history/{session}.json)
        │   ├─→ save(), load(), export()
        │   ├─→ list_sessions()
        │   └─→ switch_session()
        │
        └─→ Agent.run() | Agent.stream()
            ├─→ provider.chat() | provider.stream()
            └─→ Return response | Yield StreamChunk
```

### Key Classes

| Class | Purpose |
|-------|---------|
| `Agent` | Main agent with run(), stream(), history management |
| `ConversationHistory` | Message storage with persistence, sessions |
| `SessionMetadata` | Session tracking (name, timestamps, message count) |
| `Config` | Pydantic configuration with env loading |
| `BaseLLMProvider` | Abstract interface for LLM providers |
| `AnthropicProvider` | Claude SDK with native streaming |
| `OpenAIProvider` | OpenAI SDK with native streaming |
| `CompatibleProvider` | OpenAI-compatible APIs (GLM-4, Kimi) |
| `StreamChunk` | Streaming response chunk (delta, content, is_final) |
| `LLMResponse` | Non-streaming response |
| `Tool` | Abstract base for tools |
| `ToolResult` | Structured result with success/content/error |

## Tools

### Available Tools

| Tool | Name | Description | Enabled by Default |
|------|------|-------------|-------------------|
| `WebSearchTool` | `web_search` | DuckDuckGo web search | Yes |
| `CalculatorTool` | `calculator` | Mathematical calculations | Yes |
| `PythonCodeTool` | `python_code` | Safe Python code execution | Yes |
| `FileReadTool` | `file_read` | Read file contents | Yes |
| `FileWriteTool` | `file_write` | Write content to files | Yes |
| `SystemCommandTool` | `system` | Execute system commands | No (security risk) |
| `WikipediaTool` | `wikipedia` | Wikipedia search | Yes |

### Tool Usage Examples

```python
# Calculator
{"tool": "calculator", "parameters": {"expression": "sqrt(16) + 2"}}

# Python code execution
{"tool": "python_code", "parameters": {"code": "print([x**2 for x in range(5)])"}}

# File read
{"tool": "file_read", "parameters": {"path": "/path/to/file.txt"}}

# File write
{"tool": "file_write", "parameters": {"path": "/path/to/file.txt", "content": "Hello!", "mode": "w"}}

# Web search
{"tool": "web_search", "parameters": {"query": "Python 3.12 release date"}}

# Wikipedia
{"tool": "wikipedia", "parameters": {"query": "Machine learning", "lang": "en"}}

# System command (if enabled)
{"tool": "system", "parameters": {"command": "ls -la", "shell": false}}
```

### Tool Configuration

Enable/disable tools via environment variables:

```bash
ENABLE_WEB_SEARCH=true
ENABLE_CALCULATOR=true
ENABLE_PYTHON_CODE=true
ENABLE_FILE_READ=true
ENABLE_FILE_WRITE=true
ENABLE_SYSTEM=false        # WARNING: Security risk - enable only if needed
ENABLE_WIKIPEDIA=true
```

### Security Notes

- **System commands** (`system` tool) are disabled by default due to security risks
- **File tools** support `allowed_dirs` parameter for path restrictions
- **Python code** tool uses a sandboxed environment with limited builtins
- System commands support `allowed_commands` list for whitelisting

## Streaming

### Usage

```python
# Streaming response
for chunk in agent.stream("Tell me a story"):
    print(chunk.delta, end="", flush=True)
```

### CLI Usage

```bash
# Single query with streaming
PYTHONPATH=. python main.py --stream --query "Tell me a story"

# Interactive mode - toggle with /stream command
PYTHONPATH=. python main.py
# Then type: /stream to toggle
```

### StreamChunk Fields

| Field | Type | Description |
|-------|------|-------------|
| `content` | str | Accumulated content so far |
| `delta` | str | New content in this chunk |
| `is_final` | bool | True if this is the final chunk |
| `tool_calls` | List[ToolCall] | Tool calls if any |

## History Persistence

### Storage Location
- Default: `~/.one_agent/history/{session_name}.json`
- Configurable via `HISTORY_STORAGE_DIR` env var

### Features
- **Auto-save**: Automatically saves after each message (configurable via `AUTO_SAVE_HISTORY`)
- **Session management**: Multiple named sessions supported
- **Session metadata**: Tracks creation time, last updated, message count, provider
- **Export formats**: JSON (full data) or TEXT (human-readable)

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/stream` | Toggle streaming on/off |
| `/save` | Manually save current history |
| `/sessions` | List all saved sessions |
| `/switch NAME` | Switch to a different session |
| `/export PATH` | Export history to file (json/text) |
| `/clear` | Clear current session history |

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Default provider: anthropic, openai, glm, kimi
DEFAULT_PROVIDER=anthropic

# Anthropic Claude
ANTHROPIC_API_KEY=your_key
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# OpenAI GPT-4
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4-turbo

# 智谱 AI (GLM-4)
GLM_API_KEY=your_key
GLM_MODEL=glm-4-plus

# 月之暗面 (Kimi)
KIMI_API_KEY=your_key
KIMI_MODEL=moonshot-v1-8k

# Settings
MAX_ITERATIONS=10
MAX_HISTORY_MESSAGES=50

# Streaming
STREAMING=false        # Enable streaming by default
STREAMING_ECHO=true   # Print streaming chunks to console

# History persistence
HISTORY_STORAGE_DIR=~/.one_agent/history
AUTO_SAVE_HISTORY=true
SESSION_NAME=default

# Tools (see Tools section for details)
ENABLE_WEB_SEARCH=true
ENABLE_CALCULATOR=true
ENABLE_PYTHON_CODE=true
ENABLE_FILE_READ=true
ENABLE_FILE_WRITE=true
ENABLE_SYSTEM=false        # WARNING: Security risk
ENABLE_WIKIPEDIA=true

VERBOSE=false
```

## Conventions

- **Provider keys**: `anthropic`, `openai`, `glm`, `kimi`
- **Tool names**: lowercase with underscores (`web_search`, `python_code`, `file_read`)
- **Message roles**: `system`, `user`, `assistant`, `tool`
- **Provider naming**: ProviderConfig with `provider`, `api_key`, `model`, `base_url`
- **Error handling**: Tools return `ToolResult(success, content, error)`
- **Model detection**: CompatibleProvider auto-detects native tool support
- **Session files**: Named JSON files in history storage directory
- **Streaming**: Use `Agent.stream()` for streaming responses
- **Tool creation**: Subclass `Tool` and implement `execute()`
