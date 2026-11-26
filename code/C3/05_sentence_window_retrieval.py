import os
import sys
from pathlib import Path

from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core import Document, VectorStoreIndex, Settings
# from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike
from pypdfium2 import PdfDocument


# 用同一份英文 PDF、同一个问题，比较两种 RAG 检索方式。
# 句子窗口检索先用单句找位置，命中后再把前后几句交给百炼。
# 常规分块检索直接用较大的文本块来搜，命中后就把这个块交给百炼。

# 窗口检索策略优势:
# 通过 精确检索小文本块（单个句子），再扩展上下文（句子窗口）的方式，
# 为大语言模型提供了高度相关且信息丰富的上下文，从而生成了质量更高的答案。


# 1. 配置模型
load_dotenv()

# 负责定位 .venv 里已有的断句数据
# Path(sys.prefix) -> PosixPath('/Users/jarvis/Code/PythonProjects/all-in-rag/.venv')
nltk_data_dir = Path(sys.prefix) / "nltk_data"
print(nltk_data_dir)    
if (nltk_data_dir / "_static" / "nltk_cache").is_dir():
    os.environ.setdefault("NLTK_DATA", str(nltk_data_dir))

Settings.llm = OpenAILike(
    model=os.environ["DASHSCOPE_MODEL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    api_base=os.environ["DASHSCOPE_BASE_URL"],
    is_chat_model=True,
    temperature=0.1,    # temperature=0.1` 让生成结果通常更稳定。
)
# Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY"))
# 用的是本地 BAAI/bge-small-en，负责把文档内容和用户问题转成向量，供相似度搜索使用。
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# 2. 加载文档
# Path(__file__).resolve().parents[0] -> PosixPath('/Users/jarvis/Code/PythonProjects/all-in-rag/code/C3')
# Path(__file__).resolve().parents[1] -> PosixPath('/Users/jarvis/Code/PythonProjects/all-in-rag/code')
# Path(__file__).resolve().parents[2] -> PosixPath('/Users/jarvis/Code/PythonProjects/all-in-rag')
# Path(__file__).resolve().parents[3] -> PosixPath('/Users/jarvis/Code/PythonProjects')
pdf_path = Path(__file__).resolve().parents[2] / "data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"
documents = []
print("正在读取 PDF 文本…", flush=True)
# `PdfDocument` 逐页提取 PDF 的文字，每个有文字的页面对应一个 Document，页码保存在 metadata["page"]。
# 所以这里的 documents 是一个列表，里面装的是各页文档。
with PdfDocument(pdf_path) as pdf:
    for page_number, page in enumerate(pdf, start=1):
        text = page.get_textpage().get_text_bounded()
        if text.strip():
            documents.append(Document(text=text, metadata={"page": page_number}))
        page.close()
print("页数 len(documents):", len(documents))

# 3. 创建节点与构建索引
# 3.1 句子窗口索引
# SentenceWindowNodeParser 把每一页拆成句子，一个句子生成一个 Node。
# 假设第 12 页有 8 句话，第 4 句是「AMOC 可能减弱」，那第 4 个节点的内容大致如下：
# node.text 第 4 句，「AMOC 可能减弱」
# node.metadata["original_text"] 第 4 句的原文，供后面查看最初命中了哪句
# node.metadata["window"] 第 1 到第 7 句连起来的文本
# node.metadata["page"] 12
# window_size=3 表示左右各取最多 3 句，加上当前句，最多共 7 句。 
# 如果命中的是页首第 1 句，就只有第 1 到第 4 句。
# 因为这份代码把每页作为一个 Document，所以窗口也以当前页为边界。
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
sentence_nodes = node_parser.get_nodes_from_documents(documents)
print(f"正在为 {len(sentence_nodes)} 个句子节点生成向量…", flush=True)

# VectorStoreIndex 建索引时，用的是单句正文，加上页码等允许参与向量化的元数据。
# `window` 和 original_text 作为额外信息保存，完整窗口留到检索命中后再用。
sentence_index = VectorStoreIndex(sentence_nodes, show_progress=True)

# 3.2 常规分块索引 (基准)
# 尽量沿句子边界组织较大的文本块。`chunk_size=512` 是分块的 token 预算。
# 建索引时，每个块生成一个向量。查询时取最相关的 2 个文本块，直接作为回答的上下文。
base_parser = SentenceSplitter(chunk_size=512)
base_nodes = base_parser.get_nodes_from_documents(documents)
print(f"正在为 {len(base_nodes)} 个常规文本块生成向量…", flush=True)
base_index = VectorStoreIndex(base_nodes, show_progress=True)

# 4. 构建查询引擎
sentence_query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        # `MetadataReplacementPostProcessor` 读出这两个节点各自的 metadata["window"]，
        # 用窗口文本替换它们交给回答阶段的正文。
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
base_query_engine = base_index.as_query_engine(similarity_top_k=2)

# 5. 执行查询并对比结果
query = "What are the concerns surrounding the AMOC?"
print(f"查询: {query}\n")

print("--- 句子窗口检索结果 ---")
window_response = sentence_query_engine.query(query)
print(f"回答: {window_response}\n")

print("--- 常规检索结果 ---")
base_response = base_query_engine.query(query)
print(f"回答: {base_response}\n")


# 调用 sentence_query_engine.query(query) 时，执行顺序是：
# 1. 搜索。BGE 把问题转成向量，从索引里找最相似的 2 个句子节点。
# 2. 扩展上下文。`MetadataReplacementPostProcessor` 读出这两个节点各自的 metadata["window"]，用窗口文本替换它们交给回答阶段的正文。
# 3. 生成回答。LlamaIndex 把问题和这两个窗口交给百炼。

