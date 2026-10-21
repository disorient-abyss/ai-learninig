import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. 加载配置并校验
load_dotenv("./keys/qwen_key.txt")

api_key = os.getenv("Qwen_API_KEY")
api_base = os.getenv("Qwen_API_BASE")

if not api_key:
    raise ValueError("未读取到 Qwen_API_KEY，请检查 ./keys/qwen_key.txt 文件路径或内容。")

# 注意：若使用官方标准模型名，常用为 "qwen-max"、"qwen-plus" 等；参数名为 dashscope_api_key
llm = ChatTongyi(
    model="qwen-plus",
    dashscope_api_key=api_key,
    temperature=0.2,
    top_p=0.8,
    enable_search=False,
    max_tokens=3000,
)

# 2. 定义支持历史上下文的 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友善、专业且乐于助人的 AI 助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm

# 4. 会话状态管理 (必须在 RunnableWithMessageHistory 之前定义)
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
    # dialogues = [
    #     "嗨，你好啊！",
    #     "你好，我是disorient，是一个学习ai的学生",
    #     "我是谁？"
    # ]

    # for user_input in dialogues:
    #     print(f"User: {user_input}")
    #     reply = chat(user_input)
    #     print(f"AI: {reply}\n" + "-" * 40)

    # 第一次调用
    user_query = "你好，我是disorient，是一个学习ai的学生"
    print(f"disorient: {user_query}")
    reply = chat(user_query)
    print(f"AI: {reply}\n" + "-" * 40)

    # 第二次调用
    user_query_2 = "我是谁？"
    print(f"disorient: {user_query_2}")
    reply_2 = chat(user_query_2)
    print(f"AI: {reply_2}\n" + "-" * 40)