import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader

# 加载环境变量
load_dotenv()

# 加载本地 markdown 文件
markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"
# loader = UnstructuredMarkdownLoader(markdown_path)
loader = TextLoader(markdown_path, encoding="utf-8")
# TextLoader 只是创建了加载器，真正读文件发生在下一行。
# load() 读取文件，把原文和来源信息装进一个 Document，再放进列表返回。
# 这个示例只读一个文件，所以 docs 里有一个元素。
docs = loader.load()
print(len(docs))                    # 1
print(docs[0].page_content[:200])   # Markdown 原文的前 200 个字符
print(docs[0].metadata)             # 包含来源文件路径

# 到这里，整篇 Markdown 还在一个 Document 里。
# 后面的 text_splitter.split_documents(docs) 才会把它切成多个小块。

# 文本分块 (Chunking)
# 第一行创建分块器，使用默认规则。第二行才真正切分。
# `docs` 和 chunks 都是 Document 列表，每个新块的 page_content 是一小段原文，也会继承原文档的 source 等元数据。
# text_splitter = RecursiveCharacterTextSplitter()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=200
)
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50
# )

# RecursiveCharacterTextSplitter 里的 “递归”，指的是某一段太长时，换一种更细的分隔方式继续切。 
# 默认尝试顺序是：["\n\n", "\n", " ", ""]，也就是优先按空行切，再试换行、空格，最后才按单个字符切。
chunks = text_splitter.split_documents(docs)

# 默认有两个重要参数：
# chunk_size=4000：一个块最多约 4000 个字符，默认按 Python 的 len() 计算。
# chunk_overlap=200：尽量让相邻块保留一小段重叠内容，避免关键描述刚好被切在边界上。实际重叠多少取决于分隔位置。
print(len(docs), len(chunks))
print(chunks[0].page_content[:200])
print(chunks[0].metadata)

# 后续这段代码把 chunks 变成一个可以按语义搜索的知识库。
# `HuggingFaceEmbeddings` 负责把文字变成向量，`InMemoryVectorStore` 负责把向量和对应的原文存起来，供后续查询。

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",            # 指定使用哪个嵌入模型
    model_kwargs={'device': 'cpu'},                 # 指定在本机 CPU 上运行
    # 把每个向量的长度缩放到 1，保留它的方向。后面用余弦相似度比较时，看的就是两个向量的方向有多接近。
    encode_kwargs={'normalize_embeddings': True}    # 启用嵌入归一化
)
# 这一步只准备好了模型。真正给 chunks 生成向量的是后面的 add_documents()。

# 构建向量存储，负责保存和查找这些向量
# 创建一个空的向量存储，并告诉它用哪个模型。
vectorstore = InMemoryVectorStore(embeddings)
# 把每个块的 page_content 交给模型生成向量，再把向量、原文和 metadata 一起存起来。前面讲过的 source 文件路径也会保留。
# 这里的 InMemory 表示数据存在当前进程的内存里，脚本退出后，下次运行会重新建库。
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
)

# 配置大语言模型

# 使用 AIHubmix
# llm = ChatOpenAI(
#     model="glm-4.7-flash-free",
#     temperature=0.7,
#     max_tokens=4096,
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://aihubmix.com/v1"
# )

# llm = ChatOpenAI(
#     model="deepseek-chat",
#     temperature=0.7,
#     max_tokens=4096,
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com"
# )

llm = ChatOpenAI(
    model=os.environ["DASHSCOPE_MODEL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    temperature=0.7,    # 控制生成的随机程度。调低后，回答通常更稳定
    max_tokens=4096,    # 本次生成设置 token 上限
)

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
# similarity_search() 会用之前的 BGE 模型把 question 转成向量，再和库里的文本块向量比较。
# `k=3` 表示最多取 3 个最相似的块。返回的 retrieved_docs 是一个 Document 列表，每个结果都带着 page_content 和 metadata。
retrieved_docs = vectorstore.similarity_search(question, k=3)
print(len(retrieved_docs))
print(retrieved_docs[0].page_content[:500])

# 将检索到的多个文本块的页面内容 (doc.page_content) 合并成一个单一的字符串，
# 并使用双换行符 ("\n\n") 分隔各个块，形成最终的上下文信息 (docs_content) 供大语言模型参考。
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# answer 是 LangChain 返回的 AIMessage 对象。
answer = llm.invoke(prompt.format(question=question, context=docs_content))

print()
print("============ answer ============")
print(answer)
print()
print(answer.content)
