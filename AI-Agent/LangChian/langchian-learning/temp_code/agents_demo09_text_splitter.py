from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


def split_text_file(
    file_path: str | Path,
    chunk_size: int = 100,
    chunk_overlap: int = 20,
    add_start_index: bool = True,
) -> list[Document]:
    """
    读取并基于字符递归分割文本文件。

    :param file_path: 待读取文本文件路径
    :param chunk_size: 每个块的最大字符长度
    :param chunk_overlap: 相邻块之间的重叠字符数
    :param add_start_index: 是否在元数据中包含块在原文本中的起始索引
    :return: 分割后的 Document 列表
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"未找到目标文件: {path}")

    content = path.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=add_start_index,
    )
    return splitter.create_documents([content])


def get_supported_languages() -> list[str]:
    """获取 RecursiveCharacterTextSplitter 支持的代码/标记语言枚举值"""
    return [lang.value for lang in Language]

def find_overlap(text1: str, text2: str) -> str:
    """找出 text1 的后缀与 text2 的前缀之间的最大重叠字符串"""
    max_overlap_len = min(len(text1), len(text2))
    for i in range(max_overlap_len, 0, -1):
        if text1.endswith(text2[:i]):
            return text2[:i]
    return ""

if __name__ == "__main__":
    # 使用相对当前脚本文件的绝对路径定位，避免从不同目录运行时找不到文件
    current_dir = Path(__file__).parent
    target_file = current_dir / "tests" / "state_of_the_union.txt"

    # 1. 文本切分演示
    try:
        docs = split_text_file(
            target_file,
            chunk_size=120,
            chunk_overlap=40,
            add_start_index=True
        )
        
        print(f"成功切分文档，总块数: {len(docs)}")
        print("=" * 60)

        # 打印前 2 块及其元数据（包含 start_index）
        for idx, doc in enumerate(docs[:2], start=1):
            print(f"[ Chunk {idx} ] (元数据: {doc.metadata})")
            print(doc.page_content)
            print("-" * 60)

        # 遍历前 3 个 Chunk，并高亮对比相邻 Chunk 之间的重叠内容
        for i in range(min(3, len(docs))):
            doc = docs[i]
            print(f"\n👉 [ Chunk {i + 1} ] (起始索引 start_index: {doc.metadata['start_index']}, 字符数: {len(doc.page_content)})")
            print(doc.page_content)

            # 如果存在下一个 Chunk，计算并展示与下一个 Chunk 的重叠区域
            if i + 1 < len(docs):
                next_doc = docs[i + 1]
                overlap_text = find_overlap(doc.page_content, next_doc.page_content)
                print("\n" + "·" * 40)
                if overlap_text.strip():
                    print(f"🔗 【与下一 Chunk 的重叠文本 (共 {len(overlap_text)} 字符)】:\n「{overlap_text.strip()}」")
                else:
                    print("🔗 【与下一 Chunk 无重叠】(按大段自然换行断开)")
                print("·" * 40)

    except FileNotFoundError as err:
        print(f"[错误] {err}")

    # 2. 支持语言列表展示
    supported_langs = get_supported_languages()
    print(f"\n支持切分的代码/标记语言共 {len(supported_langs)} 种:")
    print(", ".join(supported_langs))