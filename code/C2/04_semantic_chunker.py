from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

# 准备本地嵌入模型
# 之前用过的中文 BGE 模型，在本机 CPU 上把文本转成向量，归一化后每个向量的长度为 1。
# 第一章用这些向量来检索相似文本，这份脚本用它们来判断哪里适合切开。
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 初始化 SemanticChunker
# 创建语义分块器，选择怎么判断断点
text_splitter = SemanticChunker(
    embeddings,
    # 也可以是 "standard_deviation", "interquartile", "gradient"
    breakpoint_threshold_type="percentile",
    # breakpoint_threshold_type="standard_deviation",
    # breakpoint_threshold_type="interquartile",
    # breakpoint_threshold_type="gradient",

    # 默认正则是 r"(?<=[.?!])\s+"，意思是：遇到英文的 .、`?`、`!`，并且后面有空白，才在那里断句。
    # 中文的 。！？ 和单独的换行都不会触发这条规则。
    sentence_split_regex=r"(?<=[。！？])\s*|(?<=[.!?])\s+",
    # 上面正则表达式修改断句规则，中文句末标点后可以直接切，英文句末标点后要有空白才切。
)


# SemanticChunker 真正开始处理一篇文档时，会做这几步：
# 1. 先断句。把全文拆成几个候选句子。最后只会在这些候选句子之间挑切分点。使用标准的句子分割规则（例如，基于句号、问号、感叹号）
# 2. 给句子带一点上下文，再转成向量。默认 buffer_size=1，所以计算第 2 句的向量时，会把「第 1 句 + 第 2 句 + 第 3 句」一起交给 BGE。这些临时组合只用来判断边界，最后仍按原句拼成块。
# 3. 比较相邻位置的向量。用 1 - 余弦相似度 算距离，距离越大，说明这两个位置的内容变化越明显。
# 4. 在距离明显偏大的位置切开。percentile 默认取这些距离的第 95 百分位当阈值，距离大于阈值才切。这个阈值是根据当前文档算出来的。


# documents 是分块前的文档列表，`docs` 是分块后的列表。两者都由 Document 对象组成。
# 每个新块的 page_content 是几句拼在一起的原文，`metadata` 保留来源路径。
loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
documents = loader.load()
docs = text_splitter.split_documents(documents)

print(f"文本被切分为 {len(docs)} 个块。\n")
print("--- 前2个块内容示例 ---")
for i, chunk in enumerate(docs[:2]):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}):\n"{chunk.page_content}"')
