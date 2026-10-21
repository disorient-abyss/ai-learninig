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

# 2. 显式定义带 @tool 的工具
@tool
def get_word_length(word: str) -> int:
    """计算并返回给定单词的字母数量。"""
    return len(word.strip())

# 3. 显式配置 Model
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url=api_base,
    temperature=0,  # 工具调用场景建议设为 0 保证稳定
)

# 4. 创建并运行 Agent
agent = create_agent(
    model=llm,
    tools=[get_word_length],
    system_prompt="你是一个非常有用的助手，擅长提供单词长度信息。",
)

res = agent.invoke({"messages": [{"role": "user", "content": "单词 education 总共有几个字母?"}]})
print(res['messages'][-1].content)

