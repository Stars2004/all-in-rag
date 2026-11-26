from pathlib import Path

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 使用和建索引时相同的模型，把新问题转成向量
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 找到之前保存的索引目录
# Path(__file__) 当前脚本的文件名变成 Path 对象
# Path(__file__).resolve() 得到当前脚本的绝对路径
# Path(__file__).resolve().parent 取它所在的文件夹
persist_dir = Path(__file__).resolve().parent / "llamaindex_index_store"

# 3. 从 JSON 文件恢复存储和索引
storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
index = load_index_from_storage(storage_context)

# 4. 搜索最相似的两个文本节点
query = "LlamaIndex 是做什么的？"
retriever = index.as_retriever(similarity_top_k=2)
results = retriever.retrieve(query)

for i, result in enumerate(results, start=1):
    print(f"结果 {i}，相似度：{result.score}")
    print(result.node.get_content())