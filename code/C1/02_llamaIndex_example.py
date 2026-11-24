import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import TokenTextSplitter

load_dotenv()

# 使用 AIHubmix
# Settings.llm = OpenAILike(
#     model="glm-4.7-flash-free",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://aihubmix.com/v1",
#     is_chat_model=True
# )

# Settings.llm = OpenAI(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://api.deepseek.com"
# )

# 配置回答模型
Settings.llm = OpenAILike(
    model=os.environ["DASHSCOPE_MODEL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    api_base=os.environ["DASHSCOPE_BASE_URL"],
    is_chat_model=True,
)

# 配置建库方式
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")
# 按 token 计算块大小和重叠，之前的 RecursiveCharacterTextSplitter 默认按字符计算。
Settings.text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=200)

# 读取 Markdown。返回的 docs 是 LlamaIndex 的文档列表。
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# `VectorStoreIndex.from_documents(docs)` 读取上面配置的分块器和嵌入模型，把文档切成小块并建立向量索引。
index = VectorStoreIndex.from_documents(docs)

# 创建查询引擎。LlamaIndex 默认取 2 个块
query_engine = index.as_query_engine()

# 框架准备的提示词模板
print("================== prompts ==================")
print(query_engine.get_prompts())
print()
qa_template = query_engine.get_prompts()["response_synthesizer:text_qa_template"]
print(qa_template.get_template(llm=Settings.llm))
print()

# 先检索相关块，再用这些原文组装提示词，最后调用百炼。
print("================== answer ==================")
print(query_engine.query("文中举了哪些例子?"))