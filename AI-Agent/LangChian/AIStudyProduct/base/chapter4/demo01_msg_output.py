import os
from http import HTTPStatus

import dashscope
from dashscope import Generation
from dotenv import load_dotenv

load_dotenv("./keys/qwen_key.txt")

api_key = os.getenv("Qwen_API_KEY")
api_base = os.getenv("Qwen_API_BASE")

if not api_key:
    raise ValueError("未读取到 Qwen_API_KEY，请检查 ./keys/qwen_key.txt 文件。")

dashscope.api_key = api_key

responses = Generation.call(
    model="qwen-turbo",
    messages=[
        {"role": "system", "content": "你是一个语言专家，帮我回答问题"},
        {"role": "user", "content": "什么是人工智能？"},
    ],
    result_format="message",
)

if responses.status_code == HTTPStatus.OK:
    print(responses.output.choices[0].message.content)
else:
    print(f"Error: {responses.code} - {responses.message}")