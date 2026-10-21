import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# 1. 加载配置并设置默认的阿里百炼兼容接口 Base URL
env_file = Path("./keys/qwen_key.txt")
if env_file.exists():
    load_dotenv(env_file)

api_key = os.getenv("Qwen_API_KEY")
# 若配置文件中未指定 API_BASE，则默认使用阿里百炼兼容模式 endpoint
api_base = os.getenv("Qwen_API_BASE")

if not api_key:
    raise ValueError("未读取到 Qwen_API_KEY，请检查 ./keys/qwen_key.txt 文件路径或内容。")

# 2. 初始化适配通义千问的客户端实例
client = OpenAI(
    api_key=api_key,
    base_url=api_base,
)


def get_embeddings(
    texts: list[str],
    model: str = "text-embedding-v3",
    dimensions: int | None = None,
    openai_client: OpenAI = client,
) -> tuple[list[list[float]], int]:
    """
    批量获取输入文本列表的 Qwen Embedding 向量。

    :param texts: 待向量化的文本列表
    :param model: 模型名称 (推荐 'text-embedding-v3'，默认 1024 维)
    :param dimensions: 自定义输出维度 (text-embedding-v3 支持 1024/768/512 等)
    :param openai_client: 已配置 Qwen Key 的客户端实例
    :return: (向量列表, 消耗的总 Token 数量)
    """
    if not texts:
        return [], 0

    request_params: dict = {"input": texts, "model": model}
    if dimensions is not None:
        request_params["dimensions"] = dimensions

    try:
        response = openai_client.embeddings.create(**request_params)

        # 提取向量数据（保持原输入顺序）
        embeddings = [item.embedding for item in response.data]
        total_tokens = response.usage.total_tokens if response.usage else 0

        return embeddings, total_tokens

    except OpenAIError as err:
        print(f"[API 错误] 生成 Qwen Embedding 失败: {err}")
        raise


if __name__ == "__main__":
    example_input = [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!",
    ]

    try:
        # 1. 默认维度调用 (text-embedding-v3 默认为 1024 维)
        embeddings, tokens_used = get_embeddings(
            texts=example_input,
            model="text-embedding-v3",
        )

        print(f"成功生成向量条数: {len(embeddings)}")
        print(f"Qwen 向量默认维度 (Dimension): {len(embeddings[0])}")
        print(f"本次消耗 Token 数: {tokens_used}")
        print("=" * 50)
        print(f"首条向量预览 (前 5 个分量): {embeddings[0][:5]}...")

        # 2. 降维调用演示 (缩减至 512 维，节省向量数据库存储)
        print("-" * 50)
        compressed_embeddings, _ = get_embeddings(
            texts=example_input,
            model="text-embedding-v3",
            dimensions=512,
        )
        print(f"降维后向量维度: {len(compressed_embeddings[0])}")

    except Exception as e:
        print(f"执行中断: {e}")