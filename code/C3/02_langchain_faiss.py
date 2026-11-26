from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 创建 -> 保存 -> 加载 -> 查询

# 1. 示例文本和嵌入模型
texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]
# 列表推导式把每句包装成一个 Document，原文放在 page_content 里。
# [
# Document(metadata={}, page_content='张三是法外狂徒'), 
# Document(metadata={}, page_content='FAISS是一个用于高效相似性搜索和密集向量聚类的库。'), 
# Document(metadata={}, page_content='LangChain是一个用于开发由语言模型驱动的应用程序的框架。')
# ]
docs = [Document(page_content=t) for t in texts]
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 2. 创建向量存储并保存到本地
# FAISS.from_documents() 先把三句原文转成三个向量，再把向量加入 FAISS 索引。
# LangChain 还会记录每个向量对应哪一条 Document。这样搜到最近的向量之后，就能找回对应的原文。
vectorstore = FAISS.from_documents(docs, embeddings)

# save_local() 会在当前工作目录下创建 faiss_index_store，里面有两个文件：
# index.faiss 向量索引，用来查找最近的向量
# index.pkl	原始 Document 和「索引位置 → 文档 ID」的映射，用来把搜索结果还原成原文
local_faiss_path = "./faiss_index_store"
vectorstore.save_local(local_faiss_path)
print(f"FAISS index has been saved to {local_faiss_path}")

# 3. 加载索引并执行查询
# 加载时需指定相同的嵌入模型，并允许反序列化
loaded_vectorstore = FAISS.load_local(
    local_faiss_path,
    embeddings,
    # allow_dangerous_deserialization=True 是允许读取 index.pkl 的开关。
    allow_dangerous_deserialization=True
)

# 执行相似性搜索
# 先给问题生成向量，再用 FAISS 找最近的一个向量，
# 最后返回对应的 Document。`k=1` 表示只取最靠前的一条。
query = "FAISS是做什么的？"
# results = loaded_vectorstore.similarity_search(query, k=1)
results = loaded_vectorstore.similarity_search(query, k=2)
# results:
# [
# Document(id='02cbfc89-2e8b-4f40-9809-ab5162df4cac', metadata={}, page_content='FAISS是一个用于高效相似性搜索和密集向量聚类的库。'), 
# Document(id='9ad18a77-3d4a-4cc3-a11a-4012d0ac0ec0', metadata={}, page_content='LangChain是一个用于开发由语言模型驱动的应用程序的框架。')
# ]

print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for doc in results:
    print(f"- {doc.page_content}")