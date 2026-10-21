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

    # 取值范围特点适用场景 (0, 2.0]
    # 0.1 ~ 0.3 确定性极强、严谨收敛、重复度高代码生成、数学计算、数据抽取、精确问答、JSON 解析
    # 0.7 ~ 1.0	表达自然、平衡准确度与多样性（默认推荐）	通用闲聊、文章改写、知识库问答、翻译
    # 1.2 ~ 1.8	随机性强、脑洞大、富有发散性	故事创作、写诗、头脑风暴、广告文案构思
    temperature = 0.1,
    max_tokens=1000,
    stream = True,
    incremental_output=True,  # 关键参数：开启增量返回，适配打字机流式打印
    result_format="message",  # 推荐：使用统一的 message 格式
)

for response in responses:
    if response.status_code == HTTPStatus.OK: 
        # 使用 message 格式时通过 choices[0].message.content 获取内容
        content = response.output.choices[0].message.content
        # flush=True：强制实时刷新控制台缓冲区，确保流式打字效果即时展现。
        print(content, end="", flush=True)  
    else:
        print(f"\nError: {response.code} - {response.message}")
        break
print()  # 结束后换行b