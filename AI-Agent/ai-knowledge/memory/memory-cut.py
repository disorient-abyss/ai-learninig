import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# 1. 加载配置并校验
load_dotenv("./keys/qwen_key.txt")

api_key = os.getenv("Qwen_API_KEY")
api_base = os.getenv("Qwen_API_BASE")

if not api_key:
    raise ValueError("未读取到 Qwen_API_KEY，请检查 ./keys/qwen_key.txt 文件路径或内容。")

llm = ChatOpenAI(
    model="qwen-plus",
    api_key=api_key,
    base_url=api_base,
    temperature=0.2,
    top_p=0.8,
    max_tokens=2000,
)

N = 2
trim_messages=trim_messages(
    max_tokens=K*2
)


# 2. 定义支持历史上下文的 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友善、专业且乐于助人的 AI 助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

conversation = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

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

    # 第一次调用：模型会回复，且底层自动记下 "disorient"
    reply_1 = chat("你好，我是disorient，是一个学习ai的学生", session_id="user_disorient")
    print("第一次AI回复: ", reply_1)
    print("-" * 40)

    # 第二次调用：直接发新问题，底层会自动带上第一轮记录
    reply_2 = chat("我是谁？", session_id="user_disorient")
    print("第二次AI回复: ", reply_2)