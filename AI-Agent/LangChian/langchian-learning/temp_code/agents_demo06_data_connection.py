from pathlib import Path

from langchain_core.documents import Document

# 1. 使用 pathlib 处理路径，基于当前文件位置定位，避免因执行目录不同导致找不到文件
file_path = Path(__file__).resolve().parent / "tests/state_of_the_union.txt"

try:
    # 2. 原生读取文件
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 3. 将读取的文本封装为 LangChain 标准 Document 对象列表
    docs = [Document(page_content=text, metadata={"source": str(file_path)})]

    # 3. 结构化格式化输出
    if docs:
        first_doc = docs[0]
        print(f"文档总数: {len(docs)}")
        print(f"文档对象类型: {type(first_doc)}")
        print(f"元数据 (Metadata): {first_doc.metadata}")
        print("-" * 40)
        print(f"前 100 字符预览:\n{first_doc.page_content[:100]}...")
    else:
        print("未加载到任何文档内容。")

except FileNotFoundError:
    print(f"错误: 找不到文件 -> {file_path.resolve()}")
except Exception as e:
    print(f"加载失败: {e}")