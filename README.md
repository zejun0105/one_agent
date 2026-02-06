# One-Agent

A modular Multi-Model Business Agent framework with full SDK integration, multi-provider support, conversation history persistence, streaming response, comprehensive tool system, and MCP (Model Context Protocol) server integration.

## Features

- **Multi-Provider Support**: Anthropic Claude, OpenAI GPT-4, 智谱 AI (GLM-4), 月之暗面 (Kimi)
- **Tool System**: Web search, calculator, Python code execution, file operations, Wikipedia, and more
- **Streaming Response**: Real-time output with `--stream` flag
- **History Persistence**: Auto-save conversations with multi-session support
- **MCP Integration**: Connect to Model Context Protocol servers (GitHub, Filesystem, PostgreSQL, Puppeteer)
- **Chinese Support**: Full Chinese documentation and error handling

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys

# Interactive mode
PYTHONPATH=. python main.py

# Single query
PYTHONPATH=. python main.py --query "Hello, world!"

# With specific provider
PYTHONPATH=. python main.py --provider glm --query "你好"

# Streaming
PYTHONPATH=. python main.py --stream --query "Tell me a story"
```

## Providers

| Provider | Identifier | Default Model |
|----------|------------|---------------|
| Anthropic Claude | `anthropic` | claude-3-5-sonnet-20241022 |
| OpenAI GPT-4 | `openai` | gpt-4-turbo |
| 智谱 AI GLM-4 | `glm` | glm-4-plus |
| 月之暗面 Kimi | `kimi` | moonshot-v1-8k |

## Tools

| Tool | Description | Enabled |
|------|-------------|---------|
| `web_search` | DuckDuckGo web search | Yes |
| `calculator` | Mathematical calculations | Yes |
| `python_code` | Safe Python execution | Yes |
| `file_read` | Read file contents | Yes |
| `file_write` | Write to files | Yes |
| `wikipedia` | Wikipedia search | Yes |
| `system` | Execute system commands | No (security) |

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Default provider
DEFAULT_PROVIDER=anthropic

# Provider API keys
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
GLM_API_KEY=your_key
KIMI_API_KEY=your_key

# Settings
MAX_ITERATIONS=10
MAX_HISTORY_MESSAGES=50
STREAMING=false
VERBOSE=false
```

## Documentation

For complete documentation, see [how-to-use.md](how-to-use.md).

## License

MIT
