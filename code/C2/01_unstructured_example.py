from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用 Unstructured 加载并解析 PDF 文档
# partition() 是通用入口，`partition_pdf()` 是专门解析 PDF 的入口。
# partition() 适合要处理 PDF、Word、HTML 等多种文件，让 Unstructured 自己选择解析器。也可以传 content_type 提示文件类型。
# partition_pdf() 适合已经知道输入是 PDF，想直接配置解析策略、OCR 语言等 PDF 专用选项。

# 方法一：使用 partition() 处理
# elements = partition(
#     filename=pdf_path,                  # 读哪个文件
#     content_type="application/pdf"      # 用 PDF 解析器
# )

# 方法二：使用 partition_pdf() 处理
# 明确处理的是 PDF
elements = partition_pdf(
    filename=pdf_path,
    # strategy="ocr_only",
    # strategy="hi_res",
    strategy="fast",
)
# ocr_only：把页面转成图片，再通过 OCR 提取文字。
# hi_res：会用版面模型识别页面上的内容区域，再提取和整理文字。

print(type(elements))

# TextLoader 把一篇 Markdown 原文放进一个 Document。
# 这里的 partition() 会尝试识别文档结构，把内容拆成一个个 Element。
# Element 可以理解成一张张带标签的卡片，每张卡片有正文、类别和可用的元数据。
first = elements[0]
print(first.category)             # 类别，例如 Header
print(str(first))                 # 这个元素的文字内容
print(first.metadata.page_number) # PDF 页码，如果解析器有提供
print([str(e) for e in elements])

# 打印解析结果
print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计元素类型
from collections import Counter
print([e.category for e in elements])
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")
# 元素类型: {'Header': 22, 'Title': 195, 'UncategorizedText': 41, 'NarrativeText': 3, 'Footer': 15, 'ListItem': 3}
# Title 表示标题
# `NarrativeText` 表示正文段落
# `Header` 和 Footer 分别是页眉和页脚
# `ListItem` 是列表项
# `UncategorizedText` 是没有归到更具体类别的文本
# 这些类别是解析器的判断。比如输出里有 195 个 Title，意思是有 195 个元素被识别为标题

# 显示所有元素
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)