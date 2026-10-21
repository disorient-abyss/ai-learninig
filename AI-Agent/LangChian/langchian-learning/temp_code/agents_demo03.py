import os

from dotenv import load_dotenv
from openai import OpenAI  # <-- 改用原生 openai 库

# 1. 加载配置并校验
load_dotenv("./keys/qwen_key.txt")

api_key = os.getenv("Qwen_API_KEY")
api_base = os.getenv("Qwen_API_BASE")

if not api_key:
    raise ValueError("未读取到 Qwen_API_KEY，请检查 ./keys/qwen_key.txt 文件路径或内容。")

client = OpenAI(
    api_key=api_key,
    base_url=api_base,
)

response = client.chat.completions.create(
    # model="qwen-deep-research-2025-12-15",
    model="qwen-plus",
    # 🌐 启用搜索
    web_search_options={},
    messages=[
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "你是一个非常有用的助手，搜索信息并计算数字。"
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "姚明的老婆的身高是多少?"
                }
            ]
        }
    ],
    temperature=0.7
)

print(response.choices[0].message.content)