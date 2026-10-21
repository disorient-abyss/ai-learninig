from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
    <head>
        <title>LangChain</title>
        <style>
            body { font-family: Arial, sans-serif; }
            h1 { color: darkblue; }
        </style>
    </head>
    <body>
        <div>
            <h1>LangChain</h1>
            <p>Building applications with LLMs through composability.</p>
        </div>
        <div>
            As an open source project in a rapidly developing field, we are extremely open to contributions.
        </div>
    </body>
</html>
"""


def split_html_content(
    html_text: str,
    chunk_size: int = 100,
    chunk_overlap: int = 20,
) -> list[Document]:
    """
    针对 HTML 语义标签（如 <body>, <div>, <p>）进行结构化递归切分。

    :param html_text: 原始 HTML 字符串
    :param chunk_size: 每个分块的目标字符上限
    :param chunk_overlap: 分块间的重叠字符数
    :return: 封装好的 Document 列表
    """
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.HTML,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.create_documents([html_text.strip()])


def split_html_file(
    file_path: str | Path,
    chunk_size: int = 100,
    chunk_overlap: int = 20,
) -> list[Document]:
    """读取本地 HTML 文件并执行结构化切分"""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"未找到 HTML 文件: {path}")

    content = path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.HTML,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # 使用 split_documents 保留文件来源元数据
    raw_doc = Document(page_content=content, metadata={"source": str(path)})
    return splitter.split_documents([raw_doc])


def find_overlap(text1: str, text2: str) -> str:
    """计算 text1 结尾与 text2 开头的最大重叠字符子串"""
    max_overlap_len = min(len(text1), len(text2))
    for i in range(max_overlap_len, 0, -1):
        if text1.endswith(text2[:i]):
            return text2[:i]
    return ""


if __name__ == "__main__":
    # 1. 查看 LangChain 针对 HTML 预设的分隔符优先级
    html_separators = RecursiveCharacterTextSplitter.get_separators_for_language(
        Language.HTML
    )
    print("HTML 语法切分优先级（按标签与换行逐级降级）:")
    print(html_separators)
    print("=" * 60)

    # 2. 切分 HTML 文本（设置 chunk_size=80 观察标签级别的断句效果）
    docs = split_html_content(SAMPLE_HTML, chunk_size=80, chunk_overlap=15)
    print(f"成功切分 HTML，总块数: {len(docs)}\n")

    # 3. 打印切分结果与重叠详情
    for i, doc in enumerate(docs):
        print(f"👉 [ Chunk {i + 1} ] (字符数: {len(doc.page_content)})")
        print(doc.page_content)

        if i + 1 < len(docs):
            overlap = find_overlap(doc.page_content, docs[i + 1].page_content)
            if overlap.strip():
                print(f"🔗 【重叠部分 ({len(overlap)} 字符)】: 「{overlap.strip()}」")
            else:
                print("🔗 【按 HTML 标签块独立断开，无字面重叠】")
        print("-" * 60)