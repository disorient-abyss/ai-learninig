import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# 1. 加载配置并校验
load_dotenv("./keys/deepseek_key.txt")

api_key = os.getenv("DEEPSEEK_API_KEY")
api_base = os.getenv("DEEPSEEK_API_BASE")

if not api_key:
    raise ValueError("未读取到 DEEPSEEK_API_KEY，请检查 ./keys/deepseek_key.txt 文件路径或内容。")

# 2. 显式配置 Model
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url=api_base,
    temperature=0,  # 工具调用场景建议设为 0 保证稳定
)

# 例如：保留最近 2 轮历史（即 4 条消息）
K = 2

# 3. 定义消息修剪器（保留最后 max_tokens，并保证 system 提示词始终保留）
trimmer = trim_messages(
    max_tokens= K * 2,                # 最大保留 Token 数，也可以通过 K * <?> 的方式记录保留几轮对话
    strategy="last",                  # 保留最近的消息（滑动窗口）
    token_counter=len,                # 使用对应模型的分词器计算精确长度
    include_system=True,              # 始终保留 system 角色指令
    start_on="human",                 # 裁剪后确保以用户提问开头，防止格式错乱
)

# 4. 定义支持历史上下文的 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = (
    RunnablePassthrough.assign(
        history=itemgetter("history") | trimmer
    )
    | prompt
    | llm
)

# 5. 会话状态管理 (基于 Session ID 隔离)
session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# 6. 构建带记忆的 Runnable 链
conversation = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 7. 对话调用函数
def chat(message: str, session_id: str = "default_user") -> str:
    response = conversation.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )
    return str(response.content)

# 8. 测试多轮上下文对话
if __name__ == "__main__":
    dialogues = [
        # 第 1 轮：植入特征信息（暗号）
        "我的幸运数字是 9527，请记住它。",
        
        # 第 2 轮：正常闲聊（占用 1 轮）
        "今天北京的天气真不错，适合出去散步。",
        
        # 第 3 轮：继续闲聊（此时第 1 轮应该已被挤出 2 轮窗口）
        "帮我推荐一本关于人工智能的书籍吧。",
        
        # 第 4 轮：验证提问
        "你还记得我的幸运数字是多少吗？"
    ]
    
    for user_input in dialogues:
        print(f"User: {user_input}")
        reply = chat(user_input)
        print(f"AI: {reply}\n" + "-" * 40)


