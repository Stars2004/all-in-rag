from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# TextLoader 用 UTF-8 读取文件。因为这里只读了一个文件，
# `docs` 里有一个 Document，`docs[0].page_content` 是全文，`docs[0].metadata` 记录了来源路径。
loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
docs = loader.load()

# 分块器会在当前文本里找第一个存在的分隔符：
# 先找空行，再找单个换行，然后依次找句号、逗号和空格。最后的 "" 表示按单个字符拆分。
text_splitter = RecursiveCharacterTextSplitter(
    # 针对中英文混合文本，定义一个更全面的分隔符列表
    separators=["\n\n", "\n", "。", "，", " ", ""], # 按顺序尝试分割
    chunk_size=200,
    chunk_overlap=10,
    keep_separator="end"    # 把分隔符保留在各个片段的末尾
)

# 执行分块，查看结果
chunks = text_splitter.split_documents(docs)

print(f"文本被切分为 {len(chunks)} 个块。\n")
print("--- 前5个块内容示例 ---")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')



# RecursiveCharacterTextSplitter() 递归字符分块
# 核心思想：
# 先粗切，只对太长的片段继续细切。
#
# 具体流程：
# 1. 按顺序找当前文本中第一个存在的分隔符，用它切开文本。
#    本例依次尝试：空行 → 换行 → 句号 → 逗号 → 空格 → 单个字符。
# 2. 遇到短片段先暂存，相邻的短片段能装进同一块就合并。
# 3. 遇到长片段，先输出前面暂存的短片段，保持原文顺序。
#    再拿这个长片段，用剩余的、更细的分隔符重复上面的过程。
# 4. 遍历结束后，把最后暂存的短片段也合并并输出。
# chunk_size 控制块的大小；chunk_overlap 尝试保留上一块尾部的一小段，
# 让相邻块的上下文接得上。实际重叠可能小于设定值。
