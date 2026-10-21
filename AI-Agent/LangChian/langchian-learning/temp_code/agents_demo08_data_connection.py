import os

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.documents import Document

# 1. 安全加载环境变量（若后续需要结合 LLM 处理，可在同目录创建 .env 文件写入 OPENAI_API_KEY）
load_dotenv()


def load_web_documents(
    urls: list[str],
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
) -> list[Document]:
    """
    使用 UnstructuredURLLoader 加载并解析指定网页列表为 Document。

    :param urls: 网页 URL 列表
    :param user_agent: 请求头 User-Agent，避免部分网站拦截
    """
    if not urls:
        return []

    # 配置请求头以应对防爬机制（如 403 Forbidden）
    headers = {"User-Agent": user_agent}
    loader = UnstructuredURLLoader(urls=urls, headers=headers)
    return loader.load()


if __name__ == "__main__":
    target_urls = ["https://react-lm.github.io/"]
    docs = load_web_documents(urls=target_urls)

    # 边界防御：检查抓取结果是否为空，避免 data[0] 引发 IndexError
    if not docs:
        print("[提示] 未抓取到任何内容或页面解析失败。")
    else:
        print(f"成功加载文档数量: {len(docs)}")
        print("=" * 50)

        first_doc = docs[0]
        print(f"来源 (Source): {first_doc.metadata.get('source', '未知')}")
        print("-" * 50)
        print("正文内容预览 (前 300 字符):")
        print(first_doc.page_content[:300].strip())