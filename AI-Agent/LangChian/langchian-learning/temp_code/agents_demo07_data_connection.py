import json
import re
import urllib.error
import urllib.request

import arxiv
import fitz  # 来自 pymupdf
from langchain_core.documents import Document

# 匹配现代 arXiv ID (如 2412.02612 / 2412.02612v1) 与旧版分类 ID (如 math/0309136)
ARXIV_ID_PATTERN = re.compile(
    r"^(\d{4}\.\d{4,5}(v\d+)?|[a-zA-Z\-]+(\.[a-zA-Z]{2})?\/\d{7})$"
)


def is_arxiv_id(query: str) -> bool:
    """精准判断输入是 arXiv ID 还是常规搜索词"""
    return bool(ARXIV_ID_PATTERN.match(query.strip()))


def extract_text_from_pdf_stream(pdf_bytes: bytes) -> str:
    """从内存中的 PDF 二进制流提取纯文本"""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n\n".join(page.get_text() for page in doc).strip()


def load_arxiv_documents(
    query: str,
    max_docs: int = 5,
    timeout: int = 30,
    headers: dict | None = None,
) -> list[Document]:
    """
    通过 arXiv API 检索论文并在内存中解析为 LangChain Document 列表。
    
    :param query: 论文 ID (如 '2412.02612v1') 或搜索关键词
    :param max_docs: 最大返回篇数（仅关键词搜索时生效）
    :param timeout: PDF 下载超时时间 (秒)
    :param headers: 自定义 HTTP 请求头
    """
    client = arxiv.Client(page_size=max_docs, delay_seconds=3.0, num_retries=3)
    req_headers = headers or {"User-Agent": "ArxivDocLoader/1.0 (Python/LangChain)"}

    # 构造检索条件
    query_clean = query.strip()
    search = (
        arxiv.Search(id_list=[query_clean])
        if is_arxiv_id(query_clean)
        else arxiv.Search(query=query_clean, max_results=max_docs)
    )

    documents: list[Document] = []
    for result in client.results(search):
        try:
            # 携带 Header 与 Timeout 避免被服务端拦截或长时间挂起
            req = urllib.request.Request(result.pdf_url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                pdf_bytes = response.read()

            text = extract_text_from_pdf_stream(pdf_bytes)

            # 遵循 LangChain 标准元数据命名规范
            doc = Document(
                page_content=text,
                metadata={
                    "source": result.pdf_url,
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "published": result.published.strftime("%Y-%m-%d"),
                    "summary": result.summary,
                    "entry_id": result.entry_id,
                    "primary_category": result.primary_category,
                    "categories": result.categories,
                },
            )
            documents.append(doc)

        except (urllib.error.URLError, TimeoutError) as net_err:
            print(f"[警告] 下载 PDF 失败 ({result.pdf_url}): {net_err}")
        except Exception as parse_err:
            print(f"[警告] 解析 PDF 失败 ({result.entry_id}): {parse_err}")

    return documents


if __name__ == "__main__":
    target_query = "2608.13558"
    docs = load_arxiv_documents(query=target_query, max_docs=5)

    if not docs:
        print(f"未检索到与 '{target_query}' 相关的论文或下载解析失败。")
    else:
        print(f"成功获取文档数量: {len(docs)}")
        print("=" * 50)

        first_doc = docs[0]
        print("元数据 (Metadata):")
        print(json.dumps(first_doc.metadata, ensure_ascii=False, indent=2))

        print("-" * 50)
        print(f"论文正文前 200 字符预览:\n{first_doc.page_content[:200].strip()}...")