
## 一、核心系统提示词（System Prompt）

```markdown
# AI Agent 系统身份

你是一个自主的 AI Agent，能够通过工具调用来完成复杂任务。

## 核心能力

1. **任务分解**：将复杂问题拆解为可执行的步骤
2. **工具使用**：根据需求选择并调用合适的工具
3. **结果综合**：整合多个工具的输出，形成最终答案
4. **错误处理**：当工具执行失败时，尝试替代方案

## 工作流程

你必须遵循以下思维链（Chain of Thought）：

### Step 1: 理解任务
- 分析用户意图
- 识别关键信息和约束条件

### Step 2: 规划执行
- 判断是否需要工具（如果可以直接回答，则不调用工具）
- 如需工具，列出执行计划

### Step 3: 执行与观察
- 调用工具并观察结果
- 如果结果不足，继续调用其他工具

### Step 4: 综合回答
- 基于工具返回的信息，生成完整答案
- 确保答案准确、完整、易懂

## 工具调用规则

1. **必须使用标准格式**调用工具（见下方格式说明）
2. **一次只调用必要的工具**，避免冗余
3. **先思考再行动**：在调用工具前，简要说明为什么需要这个工具
4. **验证结果**：检查工具返回是否符合预期

## 响应格式

### 当需要调用工具时：
```
【思考】简要说明为什么需要调用此工具
【行动】调用工具
```

### 当给出最终答案时：
```
【结论】基于工具结果的完整答案
```

## 注意事项

- 如果工具返回错误，尝试调整参数重试或使用替代方案
- 不要编造工具不存在的功能
- 保持简洁，避免冗余的工具调用
```

---

## 二、工具定义提示词模板

你可以使用以下工具来完成任务。每个工具都有特定的功能和参数要求。

### 1. web_search
**功能**：在互联网上搜索实时信息
**使用场景**：需要最新资讯、实时数据、不在训练数据中的信息
**参数**：
- query (string, 必需): 搜索关键词
- num_results (integer, 可选): 返回结果数量，默认 5

**示例**：
```json
{
  "tool": "web_search",
  "parameters": {
    "query": "2024年上海天气预报",
    "num_results": 3
  }
}
```

### 2. code_executor
**功能**：执行 Python 代码并返回结果
**使用场景**：数学计算、数据处理、算法实现
**参数**：
- code (string, 必需): 要执行的 Python 代码
- timeout (integer, 可选): 超时时间（秒），默认 30

**示例**：
```json
{
  "tool": "code_executor",
  "parameters": {
    "code": "import math\nresult = math.sqrt(144)\nprint(result)"
  }
}
```

### 3. memory_search
**功能**：从长期记忆中检索相关信息
**使用场景**：查找历史对话、用户偏好、过往决策
**参数**：
- query (string, 必需): 搜索查询
- top_k (integer, 可选): 返回结果数量，默认 3

**示例**：
```json
{
  "tool": "memory_search",
  "parameters": {
    "query": "用户的咖啡偏好",
    "top_k": 3
  }
}
```

### 4. file_reader
**功能**：读取文件内容
**使用场景**：分析文档、提取信息
**参数**：
- file_path (string, 必需): 文件路径
- encoding (string, 可选): 文件编码，默认 utf-8

### 5. memory_store
**功能**：将重要信息存储到长期记忆
**使用场景**：保存用户偏好、关键决策、重要信息
**参数**：
- content (string, 必需): 要存储的内容
- category (string, 必需): 分类标签（如 "user_preference", "decision", "fact"）

---

## 工具调用协议

根据接入的模型不同，使用对应的调用格式：

### OpenAI / GLM-4 格式（Function Calling）
模型会自动生成 tool_calls 结构，你无需手动构造

### Anthropic Claude 格式（Tool Use）
模型会自动生成 tool_use 块，你无需手动构造

### 通用格式（适用于不支持原生工具调用的模型）
使用 JSON 代码块标记：

```tool_call
{
  "tool": "工具名称",
  "parameters": {
    "参数名": "参数值"
  }
}
```

系统会解析这个 JSON 并执行对应工具。


## 三、多模型适配架构代码

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
import re

# ============ 统一的消息格式 ============
class Message:
    def __init__(self, role: str, content: str, 
                 tool_calls: Optional[List[Dict]] = None,
                 tool_call_id: Optional[str] = None):
        self.role = role  # system/user/assistant/tool
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id

# ============ 抽象 LLM 接口 ============
class LLMProvider(ABC):
    """统一的 LLM 提供商接口"""
    
    @abstractmethod
    def chat(self, messages: List[Message], 
             tools: Optional[List[Dict]] = None,
             **kwargs) -> Message:
        """发送对话请求"""
        pass
    
    @abstractmethod
    def format_tools(self, tools: List[Dict]) -> Any:
        """将工具定义转换为该模型的格式"""
        pass
    
    @abstractmethod
    def parse_response(self, response: Any) -> Message:
        """解析模型响应为统一格式"""
        pass


# ============ OpenAI 适配器 ============
class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4-turbo",
                 base_url: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def chat(self, messages: List[Message], 
             tools: Optional[List[Dict]] = None,
             **kwargs) -> Message:
        # 转换消息格式
        openai_messages = self._convert_messages(messages)
        
        # 构建请求参数
        params = {
            "model": self.model,
            "messages": openai_messages,
            **kwargs
        }
        
        if tools:
            params["tools"] = self.format_tools(tools)
            params["tool_choice"] = "auto"
        
        # 调用 API
        response = self.client.chat.completions.create(**params)
        return self.parse_response(response)
    
    def format_tools(self, tools: List[Dict]) -> List[Dict]:
        """OpenAI Function Calling 格式"""
        return [{
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
        } for tool in tools]
    
    def parse_response(self, response) -> Message:
        message = response.choices[0].message
        
        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = [{
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments)
            } for tc in message.tool_calls]
        
        return Message(
            role="assistant",
            content=message.content or "",
            tool_calls=tool_calls
        )
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict]:
        result = []
        for msg in messages:
            openai_msg = {"role": msg.role, "content": msg.content}
            
            if msg.tool_calls:
                openai_msg["tool_calls"] = [{
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"], ensure_ascii=False)
                    }
                } for tc in msg.tool_calls]
            
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id
            
            result.append(openai_msg)
        return result


# ============ Anthropic 适配器 ============
class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def chat(self, messages: List[Message], 
             tools: Optional[List[Dict]] = None,
             **kwargs) -> Message:
        # 提取 system 消息
        system_msg = None
        user_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                user_messages.append(msg)
        
        # 转换消息格式
        anthropic_messages = self._convert_messages(user_messages)
        
        # 构建请求参数
        params = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": anthropic_messages
        }
        
        if system_msg:
            params["system"] = system_msg
        
        if tools:
            params["tools"] = self.format_tools(tools)
        
        # 调用 API
        response = self.client.messages.create(**params)
        return self.parse_response(response)
    
    def format_tools(self, tools: List[Dict]) -> List[Dict]:
        """Anthropic Tool Use 格式"""
        return [{
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"]
        } for tool in tools]
    
    def parse_response(self, response) -> Message:
        content_text = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input
                })
        
        return Message(
            role="assistant",
            content=content_text,
            tool_calls=tool_calls if tool_calls else None
        )
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict]:
        result = []
        for msg in messages:
            if msg.role == "tool":
                # Anthropic 的工具结果格式
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content
                    }]
                })
            elif msg.tool_calls:
                # 有工具调用的 assistant 消息
                content_blocks = []
                if msg.content:
                    content_blocks.append({
                        "type": "text",
                        "text": msg.content
                    })
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"]
                    })
                result.append({
                    "role": "assistant",
                    "content": content_blocks
                })
            else:
                result.append({
                    "role": msg.role,
                    "content": msg.content
                })
        return result


# ============ 通用适配器（支持 GLM/Kimi 等兼容 OpenAI 的模型）============
class CompatibleProvider(OpenAIProvider):
    """
    兼容 OpenAI SDK 的模型提供商
    适用于：GLM-4, Kimi, DeepSeek 等
    """
    def __init__(self, api_key: str, model: str, base_url: str):
        super().__init__(api_key, model, base_url)
        self.supports_native_tools = self._check_tool_support()
    
    def _check_tool_support(self) -> bool:
        """检测模型是否支持原生工具调用"""
        # 这里可以维护一个支持列表
        native_support_models = [
            "glm-4",
            "glm-4-plus", 
            "moonshot-v1"  # Kimi
        ]
        return any(m in self.model.lower() for m in native_support_models)
    
    def chat(self, messages: List[Message], 
             tools: Optional[List[Dict]] = None,
             **kwargs) -> Message:
        if not self.supports_native_tools and tools:
            # 降级到文本模式工具调用
            return self._chat_with_text_tools(messages, tools, **kwargs)
        else:
            return super().chat(messages, tools, **kwargs)
    
    def _chat_with_text_tools(self, messages: List[Message],
                              tools: List[Dict], **kwargs) -> Message:
        """使用文本格式的工具调用（降级方案）"""
        # 将工具定义添加到 system prompt
        tool_descriptions = self._format_tools_as_text(tools)
        
        # 修改消息
        enhanced_messages = messages.copy()
        for msg in enhanced_messages:
            if msg.role == "system":
                msg.content += f"\n\n{tool_descriptions}"
                break
        else:
            enhanced_messages.insert(0, Message(
                role="system",
                content=tool_descriptions
            ))
        
        # 调用模型
        response = super().chat(enhanced_messages, tools=None, **kwargs)
        
        # 解析文本中的工具调用
        parsed = self._parse_text_tool_calls(response.content)
        if parsed:
            response.tool_calls = parsed
        
        return response
    
    def _format_tools_as_text(self, tools: List[Dict]) -> str:
        """将工具定义格式化为文本"""
        text = "# 可用工具\n\n"
        for tool in tools:
            text += f"## {tool['name']}\n"
            text += f"{tool['description']}\n\n"
            text += "参数：\n```json\n"
            text += json.dumps(tool['parameters'], indent=2, ensure_ascii=False)
            text += "\n```\n\n"
        
        text += """
使用工具时，请使用以下格式：
```tool_call
{
  "tool": "工具名称",
  "parameters": {参数对象}
}
"""
        return text
    
    def _parse_text_tool_calls(self, content: str) -> Optional[List[Dict]]:
        """从文本中解析工具调用"""
        pattern = r'```tool_call\s*\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if not matches:
            return None
        
        tool_calls = []
        for match in matches:
            try:
                call_data = json.loads(match)
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "name": call_data["tool"],
                    "arguments": call_data["parameters"]
                })
            except json.JSONDecodeError:
                continue
        
        return tool_calls if tool_calls else None


# ============ 工具管理器 ============
class Tool:
    def __init__(self, name: str, description: str,
                 parameters: Dict, func: callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
    
    def execute(self, **kwargs) -> str:
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {str(e)}"
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


# ============ Agent 核心 ============
class Agent:
    def __init__(self, llm_provider: LLMProvider, 
                 tools: List[Tool],
                 system_prompt: str):
        self.llm = llm_provider
        self.tools = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.messages: List[Message] = []
        
        # 初始化系统消息
        self.messages.append(Message(
            role="system",
            content=system_prompt
        ))
    
    def run(self, user_input: str, max_iterations: int = 5) -> str:
        """执行 Agent 主循环"""
        # 添加用户消息
        self.messages.append(Message(
            role="user",
            content=user_input
        ))
        
        for iteration in range(max_iterations):
            print(f"\n{'='*50}")
            print(f"迭代 {iteration + 1}/{max_iterations}")
            print(f"{'='*50}")
            
            # 调用 LLM
            response = self.llm.chat(
                messages=self.messages,
                tools=[t.to_dict() for t in self.tools.values()]
            )
            
            self.messages.append(response)
            
            # 检查是否有工具调用
            if response.tool_calls:
                print(f"\n🔧 检测到 {len(response.tool_calls)} 个工具调用")
                
                # 执行所有工具
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["arguments"]
                    
                    print(f"\n执行工具: {tool_name}")
                    print(f"参数: {json.dumps(tool_args, ensure_ascii=False)}")
                    
                    # 执行工具
                    if tool_name in self.tools:
                        result = self.tools[tool_name].execute(**tool_args)
                    else:
                        result = f"错误：工具 '{tool_name}' 不存在"
                    
                    print(f"结果: {result[:200]}...")
                    
                    # 添加工具结果消息
                    self.messages.append(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call["id"]
                    ))
            else:
                # 没有工具调用，返回最终答案
                print(f"\n✓ Agent 完成任务")
                return response.content
        
        return "达到最大迭代次数，任务未完成"
    
    def reset(self):
        """重置对话历史"""
        self.messages = [Message(
            role="system",
            content=self.system_prompt
        )]


# ============ 使用示例 ============
if __name__ == "__main__":
    # 定义工具
    def search_web(query: str) -> str:
        return f"搜索结果：关于'{query}'的最新信息..."
    
    def calculate(expression: str) -> str:
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    tools = [
        Tool(
            name="web_search",
            description="在互联网上搜索实时信息",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            },
            func=search_web
        ),
        Tool(
            name="calculator",
            description="执行数学计算",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2+2' 或 'math.sqrt(16)'"
                    }
                },
                "required": ["expression"]
            },
            func=calculate
        )
    ]
    
    # 系统提示词（使用前面设计的）
    system_prompt = """你是一个自主的 AI Agent，能够通过工具调用来完成复杂任务。

工作流程：
1. 理解用户需求
2. 判断是否需要工具（能直接回答就不调用）
3. 调用必要的工具
4. 综合信息给出答案

注意：
- 先思考再行动
- 一次只调用必要的工具
- 基于工具结果回答，不要编造信息
"""
    
    # ========== 示例 1: 使用 OpenAI ==========
    print("\n" + "="*60)
    print("示例 1: OpenAI GPT-4")
    print("="*60)
    
    openai_provider = OpenAIProvider(
        api_key="your-openai-key",
        model="gpt-4-turbo"
    )
    agent1 = Agent(openai_provider, tools, system_prompt)
    result1 = agent1.run("帮我搜索今天的天气，然后计算 25 * 4")
    print(f"\n最终答案:\n{result1}")
    
    
    # ========== 示例 2: 使用 Anthropic Claude ==========
    print("\n" + "="*60)
    print("示例 2: Anthropic Claude")
    print("="*60)
    
    anthropic_provider = AnthropicProvider(
        api_key="your-anthropic-key",
        model="claude-3-5-sonnet-20241022"
    )
    agent2 = Agent(anthropic_provider, tools, system_prompt)
    result2 = agent2.run("计算 100 的平方根")
    print(f"\n最终答案:\n{result2}")
    
    
    # ========== 示例 3: 使用 GLM-4（智谱 AI）==========
    print("\n" + "="*60)
    print("示例 3: GLM-4")
    print("="*60)
    
    glm_provider = CompatibleProvider(
        api_key="your-glm-key",
        model="glm-4-plus",
        base_url="https://open.bigmodel.cn/api/paas/v4"
    )
    agent3 = Agent(glm_provider, tools, system_prompt)
    result3 = agent3.run("搜索 Python 最新版本")
    print(f"\n最终答案:\n{result3}")
    
    
    # ========== 示例 4: 使用 Kimi（月之暗面）==========
    print("\n" + "="*60)
    print("示例 4: Kimi")
    print("="*60)
    
    kimi_provider = CompatibleProvider(
        api_key="your-kimi-key",
        model="moonshot-v1-32k",
        base_url="https://api.moonshot.cn/v1"
    )
    agent4 = Agent(kimi_provider, tools, system_prompt)
    result4 = agent4.run("帮我计算 15% 的税后是多少，假设原价 200 元")
    print(f"\n最终答案:\n{result4}")
```

---

## 四、配置文件示例

```yaml
# config.yaml
models:
  openai:
    api_key: "sk-xxx"
    model: "gpt-4-turbo"
    base_url: null
  
  anthropic:
    api_key: "sk-ant-xxx"
    model: "claude-3-5-sonnet-20241022"
  
  glm:
    api_key: "xxx.xxx"
    model: "glm-4.7"
    base_url: "https://open.bigmodel.cn/api/paas/v4"
  
  kimi:
    api_key: "sk-xxx"
    model: "moonshot-v1-32k"
    base_url: "https://api.moonshot.cn/v1"
  
  deepseek:
    api_key: "sk-xxx"
    model: "deepseek-chat"
    base_url: "https://api.deepseek.com"

agent:
  max_iterations: 5
  temperature: 0.7
  timeout: 30
```

---

## 五、使用工厂模式创建 Provider

```python
import yaml
from typing import Dict

class LLMFactory:
    @staticmethod
    def create(provider_name: str, config: Dict) -> LLMProvider:
        """根据配置创建 LLM Provider"""
        
        if provider_name == "openai":
            return OpenAIProvider(
                api_key=config["api_key"],
                model=config["model"],
                base_url=config.get("base_url")
            )
        
        elif provider_name == "anthropic":
            return AnthropicProvider(
                api_key=config["api_key"],
                model=config["model"]
            )
        
        elif provider_name in ["glm", "kimi", "deepseek"]:
            return CompatibleProvider(
                api_key=config["api_key"],
                model=config["model"],
                base_url=config["base_url"]
            )
        
        else:
            raise ValueError(f"不支持的 provider: {provider_name}")

# 使用示例
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# 动态切换模型
provider = LLMFactory.create("glm", config["models"]["glm"])
agent = Agent(provider, tools, system_prompt)
```

---

## 六、关键设计要点总结

### ✅ **统一抽象层**
- 所有模型通过 `LLMProvider` 接口访问
- 消息格式统一为 `Message` 类
- 工具定义统一为标准 JSON Schema

### ✅ **降级策略**
- 原生支持工具调用 → 直接使用
- 不支持 → 降级到文本格式工具调用

### ✅ **提示词分层**
1. **系统层**：定义 Agent 身份和工作流程
2. **工具层**：动态注入可用工具列表
3. **任务层**：用户具体请求

### ✅ **可扩展性**
- 新增模型：继承 `LLMProvider` 实现适配器
- 新增工具：创建 `Tool` 实例并注册
- 新增功能：在 `Agent` 类中扩展

