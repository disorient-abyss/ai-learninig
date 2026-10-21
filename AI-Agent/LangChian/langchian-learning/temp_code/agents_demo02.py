import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 1. 加载配置并校验
load_dotenv("./keys/deepseek_key.txt")

api_key = os.getenv("DEEPSEEK_API_KEY")
api_base = os.getenv("DEEPSEEK_API_BASE")

if not api_key:
    raise ValueError("未读取到 DEEPSEEK_API_KEY，请检查 ./keys/deepseek_key.txt 文件路径或内容。")

# 这里定义一个无效的工具，工具还是启作用的 
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# 3. 显式配置 Model
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url=api_base,
    temperature=0,  # 工具调用场景建议设为 0 保证稳定
)

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="你是一个非常有用的助手，擅长提供城市天气信息。",
)

# Run the agent
res = agent.invoke(
    {"messages": [{"role": "user", "content": "上海的天气怎么样?"}]}
)
print(res['messages'][-1].content)