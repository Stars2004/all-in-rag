from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 用 LlamaIndex 把文本转成向量索引，再把索引保存成 JSON 文件。
# 它对应你刚学的 FAISS 示例里的「建库 → 保存」两步。

# 1. 配置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 创建示例文档
texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。"
]

# 把三句话包装成文档, 对应 LangChain 里的 Document(page_content=t)
docs = [Document(text=t) for t in texts]

# 3. 创建索引并持久化到本地
# 把文档处理成可检索的节点，再用 BGE 给节点生成向量，并记录向量和原文的对应关系。
index = VectorStoreIndex.from_documents(docs)
persist_path = "./llamaindex_index_store"
# storage_context 可以理解成管理索引所需数据的对象。
# 保存时，LlamaIndex 会把原文、向量和索引结构分别写入文件。
index.storage_context.persist(persist_dir=persist_path)
print(f"LlamaIndex 索引已保存至: {persist_path}")

# 主要有这三个：
# docstore.json 节点的原文和元数据
# default__vector_store.json 3 个向量，以及向量和节点的对应信息。打开后可以看 embedding_dict
# index_store.json 索引结构和它引用的节点 ID

