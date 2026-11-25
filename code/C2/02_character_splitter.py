from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 1. 文档加载
loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
docs = loader.load()
# docs 是一个只有一个元素的列表，`docs[0].page_content` 就是全文

# 2. 初始化固定大小分块器
# CharacterTextSplitter 默认用 `\n\n` 作为唯一的分隔符。
# 它先按这个分隔符切出段落，再依次合并短段落。如果加上下一个段落会超过 chunk_size=200 个字符，就先结束当前块。
# chunk_overlap=10 表示切换到下一个块时，尽量从上一块尾部保留不超过 10 个字符的完整片段，让上下文接得上。
text_splitter = CharacterTextSplitter(
    chunk_size=200,    # 每个块的大小
    chunk_overlap=10   # 块之间的重叠大小
)

# 3. 执行分块
# chunks 仍然是 list[Document]。每块的 page_content 是一段原文，`metadata` 继承来源文件等信息。
chunks = text_splitter.split_documents(docs)
print(type(chunks), len(chunks))
# print(chunks[0], chunks[0].page_content)
# print(chunks[1], chunks[1].page_content)
# print(chunks[2], chunks[2].page_content)

# 4. 打印结果
print(f"文本被切分为 {len(chunks)} 个块。\n")
print("--- 前5个块内容示例 ---")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 60)
    # chunk 是一个 Document 对象，需要访问它的 .page_content 属性来获取文本
    print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')
