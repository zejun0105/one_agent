# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

One-Agent is a modular Multi-Model Business Agent framework with full SDK integration and support for multiple LLM providers.

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
python main.py

# Single query
python main.py --query "What is 25 * 4?"

# Use specific provider
python main.py --provider openai --query "Hello"

# With verbose output
python main.py --verbose --query "Search for latest AI news"

# List configured providers
python main.py --list-providers

# Run tests
pytest tests/ -v
```

## Architecture

### Module Structure

```
one-agent/
├── main.py              # Entry point, CLI, agent factory
├── core/                # Core agent logic
│   ├── __init__.py      # Exports: Agent, Config, ConversationHistory
│   ├── agent.py         # Main Agent class with tool execution
│   ├── config.py        # Configuration with Pydantic models
│   └── history.py       # Conversation history management
├── providers/           # LLM provider implementations
│   ├── __init__.py      # Exports: BaseLLMProvider, AnthropicProvider, OpenAIProvider, CompatibleProvider
│   ├── base.py          # Abstract base class (ToolCall, LLMResponse, BaseLLMProvider)
│   ├── anthropic.py     # Anthropic Claude provider
│   ├── openai.py        # OpenAI GPT-4 provider
│   └── compatible.py    # GLM-4, Kimi (with native + text fallback)
├── tools/               # Tool implementations
│   ├── __init__.py      # Exports: Tool, ToolResult, WebSearchTool, CalculatorTool
│   ├── base.py          # Abstract Tool class
│   ├── web_search.py    # DuckDuckGo search tool
│   └── calculator.py    # Math expression evaluator
└── tests/               # Test suite
```

### Component Interaction

```
CLI (main.py)
    │
    ├─→ Config.load() ──→ .env
    │
    ├─→ create_agent()
    │   ├─→ ProviderFactory.create_provider()
    │   │   └─→ AnthropicProvider / OpenAIProvider / CompatibleProvider
    │   │
    │   └─→ create_tools()
    │       └─→ WebSearchTool / CalculatorTool
    │
    └─→ Agent.run(user_input)
        │
        ├─→ ConversationHistory.get_messages()
        │
        ├─→ provider.chat(messages, tools)
        │   └─→ API call to LLM
        │
        ├─→ Tool.execute() for each tool_call
        │
        └─→ Return response
```

### Key Classes

- **Agent**: Main agent class managing conversation loop and tool execution
- **ConversationHistory**: Message history with persistence support
- **Config**: Pydantic-based configuration management
- **BaseLLMProvider**: Abstract interface for LLM providers
- **Tool**: Abstract base class for tools

## Configuration

Set API keys in `.env`:

```bash
# Provider selection
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
ENABLE_WEB_SEARCH=true
ENABLE_CALCULATOR=true
VERBOSE=false
```

## Conventions

- **Provider naming**: `anthropic`, `openai`, `glm`, `kimi`
- **Tool names**: lowercase with underscores (e.g., `web_search`, `calculator`)
- **Message roles**: `system`, `user`, `assistant`, `tool`
- **Configuration**: Use Pydantic models for type safety
- **Error handling**: Return `ToolResult` with `success` flag
