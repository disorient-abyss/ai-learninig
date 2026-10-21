import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# 1. 加载配置并校验
load_dotenv("./keys/deepseek_key.txt")

api_key = os.getenv("DEEPSEEK_API_KEY")
api_base = os.getenv("DEEPSEEK_API_BASE")

if not api_key:
    raise ValueError("未读取到 DEEPSEEK_API_KEY，请检查 ./keys/deepseek_key.txt 文件路径或内容。")

# 3. 显式配置 Model
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url=api_base,
    temperature=0,  # 工具调用场景建议设为 0 保证稳定
)

# 3. 定义支持历史上下文的 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友善、专业且乐于助人的 AI 助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm

# 4. 会话状态管理 (基于 Session ID 隔离)
session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# 5. 构建带记忆的 Runnable 链
conversation = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 6. 对话调用函数
def chat(message: str, session_id: str = "default_user") -> str:
    response = conversation.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )
    return str(response.content)

# 7. 测试多轮上下文对话
if __name__ == "__main__":
    dialogues = [
        "嗨，你好啊！",
        "我很好，正在和AI进行对话呢！",
        "给我讲讲关于你的事情吧，顺便提一下我上一句话说了什么？"
    ]

    for user_input in dialogues:
        print(f"User: {user_input}")
        reply = chat(user_input)
        print(f"AI: {reply}\n" + "-" * 40)