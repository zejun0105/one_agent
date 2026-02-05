# One-Agent 使用指南

One-Agent 是一个模块化的多模型商业智能代理框架，支持多种 LLM 提供商、完整的工具系统、流式输出、对话历史持久化，以及 MCP (Model Context Protocol) 服务器集成。

## 目录

- [快速开始](#快速开始)
- [安装配置](#安装配置)
- [运行模式](#运行模式)
- [交互模式命令](#交互模式命令)
- [工具使用](#工具使用)
- [MCP 集成](#mcp-集成)
- [历史管理](#历史管理)
- [高级配置](#高级配置)
- [常见问题](#常见问题)

---

## 快速开始

### 1. 克隆并安装依赖

```bash
cd one-agent
pip install -r requirements.txt
```

### 2. 配置 API 密钥

```bash
# 复制示例配置
cp .env.example .env

# 编辑 .env 文件，填入你的 API 密钥
nano .env
```

### 3. 运行代理

```bash
# 交互模式
PYTHONPATH=. python main.py

# 单次查询
PYTHONPATH=. python main.py --query "你好，请做自我介绍"
```

---

## 安装配置

### 环境要求

- Python 3.10+
- pip

### 依赖安装

```bash
pip install -r requirements.txt
```

### API 密钥配置

编辑 `.env` 文件：

```bash
# 默认提供商
DEFAULT_PROVIDER=anthropic

# Anthropic Claude
ANTHROPIC_API_KEY=your_key_here

# OpenAI GPT-4
OPENAI_API_KEY=your_key_here

# 智谱 AI (GLM-4)
GLM_API_KEY=your_key_here

# 月之暗面 (Kimi)
KIMI_API_KEY=your_key_here
```

### 可用提供商

| 提供商 | 标识符 | 默认模型 |
|--------|--------|----------|
| Anthropic Claude | `anthropic` | claude-3-5-sonnet-20241022 |
| OpenAI GPT-4 | `openai` | gpt-4-turbo |
| 智谱 AI GLM-4 | `glm` | glm-4-plus |
| 月之暗面 Kimi | `kimi` | moonshot-v1-8k |

---

## 运行模式

### 交互模式

进入对话式交互界面：

```bash
PYTHONPATH=. python main.py
```

输出示例：

```
============================================================
  One-Agent - Multi-Model Business Agent
============================================================

Agent initialized: <core.agent.Agent object at 0x...>
Provider: anthropic (claude-3-5-sonnet-20241022)
History: 0 messages (auto-save: True)
Storage: ~/.one_agent/history/default.json
Streaming: disabled
Available tools: ['web_search', 'calculator', 'python_code', ...]

Commands:
  /help       - Show this help
  /reset      - Reset conversation
  /save       - Save history manually
  /sessions   - List all sessions
  /switch NAME - Switch to a different session
  /export PATH - Export history to file (json or text)
  /clear      - Clear current session history
  /stream     - Toggle streaming on/off
  quit        - Exit
------------------------------------------------------------

You:
```

### 单次查询模式

直接运行单个查询并退出：

```bash
# 使用默认提供商
PYTHONPATH=. python main.py --query "25 * 4 等于多少？"

# 指定提供商
PYTHONPATH=. python main.py --provider openai --query "你好"

# 带流式输出
PYTHONPATH=. python main.py --stream --query "讲一个笑话"
```

### 命令行参数

| 参数 | 简写 | 说明 |
|------|------|------|
| `--query` | `-q` | 单次查询字符串 |
| `--provider` | `-p` | 指定 LLM 提供商 |
| `--verbose` | `-v` | 显示详细输出 |
| `--stream` | `-s` | 启用流式输出 |
| `--env` | - | 指定环境文件路径 |

### 信息查询命令

```bash
# 列出已配置的所有提供商
PYTHONPATH=. python main.py --list-providers

# 列出所有会话
PYTHONPATH=. python main.py --list-sessions

# 列出 MCP 服务器
PYTHONPATH=. python main.py --list-mcp-servers
```

---

## 交互模式命令

在交互模式中，输入 `/` 开头的命令：

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助信息 |
| `/reset` | 重置当前对话 |
| `/save` | 手动保存历史 |
| `/sessions` | 列出所有会话 |
| `/switch NAME` | 切换到指定会话 |
| `/export PATH [--format json\|text]` | 导出历史到文件 |
| `/clear` | 清除当前会话 |
| `/stream` | 切换流式输出开关 |
| `/mcp` | 显示 MCP 服务器信息 |
| `quit` / `exit` / `q` | 退出 |

### 使用示例

```bash
# 在交互模式中
You: /help

You: /sessions
Sessions in ~/.one_agent/history:
  1. default (current)
     Messages: 5, Updated: 2024-01-15 10:30:00
  2. project-alpha
     Messages: 12, Updated: 2024-01-14 16:45:00

You: /switch project-alpha
Switched to session: project-alpha

You: /export ./chat_history.txt
History exported to: ./chat_history.txt

You: /stream
Streaming enabled.

You: quit
Goodbye!
```

---

## 工具使用

### 内置工具

| 工具名称 | 功能 | 默认启用 |
|----------|------|----------|
| `web_search` | DuckDuckGo 网页搜索 | 是 |
| `calculator` | 数学计算 | 是 |
| `python_code` | Python 代码执行 | 是 |
| `file_read` | 读取文件内容 | 是 |
| `file_write` | 写入文件 | 是 |
| `wikipedia` | Wikipedia 搜索 | 是 |
| `system` | 执行系统命令 | 否 (安全风险) |

### 工具使用示例

#### 网页搜索

```
You: 搜索关于 Python 3.12 发布的信息
Agent: 根据搜索结果，Python 3.12 于 2023年10月2日发布...
```

#### 数学计算

```
You: 计算 sqrt(144) + 25 * 4
Agent: 计算结果: 144 + 100 = 244
```

#### Python 代码执行

```
You: 用 Python 生成斐波那契数列前10个数字
Agent: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
```

#### 文件读取

```
You: 读取 /path/to/file.txt 的内容
Agent: [文件内容]
```

#### 文件写入

```
You: 将 "Hello, World!" 写入 /tmp/test.txt
Agent: 文件写入成功: /tmp/test.txt
```

#### Wikipedia 搜索

```
You: 搜索 Wikipedia 上关于人工智能的信息
Agent: [Wikipedia 摘要]
```

#### 系统命令 (需手动启用)

```bash
# 在 .env 中启用
ENABLE_SYSTEM=true
```

```
You: 执行 ls -la /home
Agent: [命令输出]
```

### 工具配置

通过环境变量启用/禁用工具：

```bash
# .env 文件
ENABLE_WEB_SEARCH=true
ENABLE_CALCULATOR=true
ENABLE_PYTHON_CODE=true
ENABLE_FILE_READ=true
ENABLE_FILE_WRITE=true
ENABLE_SYSTEM=false        # 注意：安全风险，默认禁用
ENABLE_WIKIPEDIA=true
```

### 安全注意事项

- **系统命令工具**默认禁用，因为存在安全风险
- **文件工具**可以配置 `allowed_dirs` 参数限制访问目录
- **Python 代码**在沙箱环境中执行，限制了内置函数
- 系统命令支持 `allowed_commands` 白名单列表

---

## MCP 集成

MCP (Model Context Protocol) 允许 One-Agent 连接外部服务器以获取额外能力。

### MCP 服务器配置

创建 `mcp_servers.json` 文件：

```json
{
  "servers": [
    {
      "name": "github",
      "command": "uvx",
      "args": ["mcp-server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      },
      "timeout": 30
    },
    {
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"],
      "timeout": 30
    },
    {
      "name": "postgres",
      "command": "uvx",
      "args": ["mcp-server-postgres", "postgresql://user:pass@localhost/db"],
      "timeout": 30
    },
    {
      "name": "puppeteer",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
      "timeout": 60
    }
  ]
}
```

### MCP 服务器配置说明

| 字段 | 说明 |
|------|------|
| `name` | 服务器唯一标识符 |
| `command` | 启动命令 (如 `uvx`, `npx`, `python`) |
| `args` | 命令参数列表 |
| `env` | 环境变量，可使用 `${VAR}` 引用外部变量 |
| `timeout` | 超时时间（秒） |

### MCP CLI 命令

```bash
# 列出配置的 MCP 服务器
PYTHONPATH=. python main.py --list-mcp-servers

# 使用自定义配置文件
PYTHONPATH=. python main.py --mcp-config /path/to/config.json --list-mcp-servers

# 连接特定服务器并查看工具
PYTHONPATH=. python main.py --mcp-connect github
```

### MCP 交互命令

在交互模式中使用：

```
You: /mcp

MCP Servers (4):
------------------------------------------------------------
  github [connected]
    Tools: 5 available
      - mcp_github_search_repositories
      - mcp_github_create_pull_request
      - mcp_github_create_issue
      - mcp_github_get_user_info
      - mcp_github_list_pull_requests

  filesystem [connected]
    Tools: 2 available
      - mcp_filesystem_read_file
      - mcp_filesystem_list_files
```

### MCP 环境变量

```bash
# .env 文件
ENABLE_MCP=true                    # 启用 MCP 集成
MCP_CONFIG_FILE=mcp_servers.json   # MCP 配置文件路径
```

### 支持的 MCP 服务器

| 服务器 | 包名 | 功能 |
|--------|------|------|
| GitHub | mcp-server-github | GitHub API 操作 |
| Filesystem | @modelcontextprotocol/server-filesystem | 文件系统访问 |
| PostgreSQL | mcp-server-postgres | 数据库查询 |
| Puppeteer | @modelcontextprotocol/server-puppeteer | 浏览器自动化 |

### MCP 工具命名

MCP 工具使用以下命名格式：`mcp_{server_name}_{tool_name}`

示例：
- `mcp_github_search_repositories`
- `mcp_filesystem_read_file`
- `mcp_postgres_query`

---

## 历史管理

### 存储位置

默认存储目录：`~/.one_agent/history/{session_name}.json`

### 特性

- **自动保存**：每条消息后自动保存（可配置）
- **多会话支持**：多个命名的会话
- **会话元数据**：记录创建时间、最后更新时间、消息数、使用的提供商
- **导出格式**：JSON（完整数据）或 TEXT（人类可读）

### 历史命令

```bash
# 列出所有会话
PYTHONPATH=. python main.py --list-sessions

# 保存当前历史（默认会话）
PYTHONPATH=. python main.py --save-history

# 保存到指定会话
PYTHONPATH=. python main.py --save-history my_project

# 加载指定会话
PYTHONPATH=. python main.py --load-history my_project

# 清除当前会话
PYTHONPATH=. python main.py --clear-history

# 清除指定会话
PYTHONPATH=. python main.py --clear-history my_project

# 导出历史
PYTHONPATH=. python main.py --export-history default ./history.json
PYTHONPATH=. python main.py --export-history default ./history.txt --format text
```

### 在交互模式中管理历史

```
You: /save
History saved to: ~/.one_agent/history/default.json

You: /sessions
Sessions in ~/.one_agent/history:
  1. default (current)
     Messages: 10, Updated: 2024-01-15 10:30:00

You: /export ./backup.txt
History exported to: ./backup.txt

You: /clear
History cleared.
```

---

## 高级配置

### 完整配置示例

```bash
# .env 文件完整示例

# =============================================================================
# Provider Selection
# =============================================================================
DEFAULT_PROVIDER=anthropic

# =============================================================================
# Anthropic Claude
# =============================================================================
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# =============================================================================
# OpenAI GPT-4
# =============================================================================
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4-turbo
OPENAI_BASE_URL=

# =============================================================================
# 智谱 AI (GLM-4)
# =============================================================================
GLM_API_KEY=your_glm_api_key
GLM_MODEL=glm-4-plus
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# =============================================================================
# 月之暗面 (Kimi)
# =============================================================================
KIMI_API_KEY=your_kimi_api_key
KIMI_MODEL=moonshot-v1-8k
KIMI_BASE_URL=https://api.moonshot.cn/v1

# =============================================================================
# Agent Settings
# =============================================================================
MAX_ITERATIONS=10
MAX_HISTORY_MESSAGES=50

# =============================================================================
# Streaming Settings
# =============================================================================
STREAMING=false
STREAMING_ECHO=true

# =============================================================================
# History Persistence
# =============================================================================
HISTORY_STORAGE_DIR=~/.one_agent/history
AUTO_SAVE_HISTORY=true
SESSION_NAME=default

# =============================================================================
# Tool Settings
# =============================================================================
ENABLE_WEB_SEARCH=true
ENABLE_CALCULATOR=true
ENABLE_PYTHON_CODE=true
ENABLE_FILE_READ=true
ENABLE_FILE_WRITE=true
ENABLE_SYSTEM=false
ENABLE_WIKIPEDIA=true

# =============================================================================
# MCP Settings
# =============================================================================
ENABLE_MCP=true
MCP_CONFIG_FILE=mcp_servers.json

# =============================================================================
# UI Settings
# =============================================================================
VERBOSE=false
COLORS=true
```

### 流式输出配置

```bash
# 默认启用流式输出
STREAMING=true

# 控制是否在控制台打印流式输出
STREAMING_ECHO=true
```

### 自定义历史存储

```bash
# 更改历史存储目录
HISTORY_STORAGE_DIR=/path/to/history

# 自动保存开关
AUTO_SAVE_HISTORY=false

# 默认会话名称
SESSION_NAME=my_session
```

---

## 常见问题

### Q1: 提示 "No provider available"

**问题**：没有配置任何 API 密钥

**解决**：在 `.env` 文件中配置至少一个提供商的 API 密钥

```bash
ANTHROPIC_API_KEY=your_key_here
```

### Q2: MCP 服务器连接失败

**问题**：MCP 服务器无法连接

**解决**：
1. 检查服务器是否正确安装
2. 验证配置文件中的命令和参数
3. 检查必要的环境变量是否设置

```bash
# 测试连接
PYTHONPATH=. python main.py --mcp-connect server_name
```

### Q3: 文件工具无法读取/写入

**问题**：文件操作失败

**解决**：
1. 检查文件路径是否正确
2. 确认有足够的权限
3. 配置 `allowed_dirs` 参数限制访问目录

### Q4: 历史记录不保存

**问题**：对话历史没有保存

**解决**：
1. 检查 `AUTO_SAVE_HISTORY` 设置
2. 确认历史存储目录存在且可写
3. 手动保存：`/save`

### Q5: 如何切换提供商

**方法 1**：通过命令行参数

```bash
PYTHONPATH=. python main.py --provider glm --query "你好"
```

**方法 2**：在交互模式中（需要代码修改支持）

### Q6: 如何添加自定义工具

参考 `tools/` 目录下的示例，创建一个继承 `Tool` 基类的工具：

```python
from tools.base import Tool, ToolResult

class MyCustomTool(Tool):
    def __init__(self):
        super().__init__(
            name="my_custom_tool",
            description="我的自定义工具",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "输入参数"}
                },
                "required": ["input"]
            }
        )

    def execute(self, **kwargs) -> ToolResult:
        # 实现工具逻辑
        return ToolResult(success=True, content="结果")
```

### Q7: 如何调试问题

启用详细输出模式：

```bash
PYTHONPATH=. python main.py --verbose --query "test"
```

这将显示：
- API 请求/响应
- 工具调用详情
- 错误堆栈跟踪

### Q8: 代理响应太慢

**可能的解决方案**：
1. 检查网络连接
2. 尝试不同的提供商
3. 减少 `MAX_HISTORY_MESSAGES` 值
4. 使用更短的 `max_tokens` 设置

---

## 示例场景

### 场景 1：代码助手

```bash
PYTHONPATH=. python main.py --provider anthropic --query "帮我用 Python 写一个快速排序算法"
```

### 场景 2：研究助手

```bash
PYTHONPATH=. python main.py --provider glm --stream --query "搜索并总结关于量子计算的最新进展"
```

### 场景 3：文件处理

```bash
PYTHONPATH=. python main.py --query "将 /home/user/data.csv 的内容读取并分析"
```

### 场景 4：多会话项目管理

```bash
# 启动交互模式
PYTHONPATH=. python main.py

# 在交互模式中
You: /switch project-alpha
You: 开始讨论项目 A 的架构设计

You: /switch project-beta
You: 开始讨论项目 B 的需求

You: /sessions
You: /export ./projects_backup.txt
```

### 场景 5：集成 GitHub MCP

```bash
# 配置 mcp_servers.json
{
  "servers": [{
    "name": "github",
    "command": "uvx",
    "args": ["mcp-server-github"],
    "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"}
  }]
}

# 在交互模式中
You: /mcp
You: 搜索我的 GitHub 仓库中包含 'feature' 的 issues
```

---

## 获取帮助

- 查看完整帮助：`python main.py --help`
- 查看 CLAUDE.md：项目架构和开发指南
- 提交 Issue：项目问题追踪

---

**版本**: 1.0.0
**最后更新**: 2024
