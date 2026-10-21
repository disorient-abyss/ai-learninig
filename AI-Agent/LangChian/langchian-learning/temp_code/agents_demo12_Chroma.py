import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# 1. 安全读取配置与密钥
load_dotenv("./keys/qwen_key.txt")
api_key = os.getenv("Qwen_API_KEY")
api_base = os.getenv("Qwen_API_BASE")

if not api_key:
    raise ValueError("未读取到 Qwen_API_KEY，请检查 ./keys/qwen_key.txt 文件。")


class QwenEmbeddings(Embeddings):
    """
    基于 OpenAI 兼容协议实现的通义千问 Embeddings 封装类。
    支持批量调用以大幅降低网络延迟与建库耗时。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "text-embedding-v3",
        dimensions: int | None = None,
        batch_size: int = 10,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量对文档列表生成向量"""
        embeddings: list[list[float]] = []
        if not texts:
            return embeddings

        # 分批次批量调用，避免单次 payload 超限，同时避免单条请求的性能瓶颈
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            params = {"input": batch, "model": self.model}
            if self.dimensions:
                params["dimensions"] = self.dimensions

            response = self.client.embeddings.create(**params)
            embeddings.extend([item.embedding for item in response.data])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """对单条查询文本生成向量"""
        params = {"input": [text], "model": self.model}
        if self.dimensions:
            params["dimensions"] = self.dimensions

        response = self.client.embeddings.create(**params)
        return response.data[0].embedding


if __name__ == "__main__":
    # 2. 跨平台安全定位测试文本路径
    current_dir = Path(__file__).resolve().parent
    target_file = current_dir / "tests" / "state_of_the_union.txt"

    if not target_file.is_file():
        raise FileNotFoundError(f"未找到目标测试文件: {target_file}")

    # 3. 实例化 Qwen 嵌入模型
    embeddings = QwenEmbeddings(
        api_key=api_key,
        base_url=api_base,
        model="text-embedding-v3",
    )

    # 4. 加载并切分文本
    raw_documents = TextLoader(str(target_file), encoding="utf-8").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    documents = text_splitter.split_documents(raw_documents)
    print(f"成功加载并切分文档，生成分块数: {len(documents)}")

    # 5. 构建内存临时 Chroma 向量数据库并向量化入库
    t0 = time.time()
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="state_of_the_union_qwen",
    )
    print(f"向量库构建完成，耗时: {time.time() - t0:.3f} 秒")
    print("=" * 60)

    query = "这里文档切片方面问了什么问题？"

    # 方式一：直接传文本进行相似度检索（自动由 embeddings.embed_query 转换为向量）
    t1 = time.time()
    docs_text_search = db.similarity_search(query, k=2)
    t1_cost = time.time() - t1

    print(f"【方式一：文本直搜】检索耗时: {t1_cost:.3f} 秒")
    print(f"Top 1 结果:\n{docs_text_search[0].page_content.strip()}")
    print("-" * 60)

    # 方式二：手动生成向量后通过向量检索（适合复杂管道预计算）
    t2 = time.time()
    query_vector = embeddings.embed_query(query)
    docs_vector_search = db.similarity_search_by_vector(query_vector, k=2)
    t2_cost = time.time() - t2

    print(f"【方式二：向量检索】向量化+检索耗时: {t2_cost:.3f} 秒")
    print(f"Top 1 结果:\n{docs_vector_search[0].page_content.strip()}")
    print("=" * 60)