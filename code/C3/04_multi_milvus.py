import os
from tqdm import tqdm
from glob import glob
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image


# 图文搜图：给它一张查询图片，再加上文字「一条龙」，它会从图库里找出最匹配的 5 张图，最后拼成一张结果图。
# 前面学过用 BGE 编码、用 @ 比较两个向量。这里仍然由 Python 生成向量，改由 Milvus 保存向量、建立索引和查找 Top 5。


# 1. 初始化设置
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"
DATA_DIR = "../../data/C3"
COLLECTION_NAME = "multimodal_demo"
MILVUS_URI = "http://localhost:19530"


# 2. 定义工具 (编码器和可视化函数)
class Encoder:
    """编码器类，用于将图像和文本编码为向量。"""
    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    # 两个方法用的是同一个模型，输出都是 768 维。
    # `model.encode(...)` 返回的是形状为 (1, 768) 的 PyTorch 张量，
    # `.tolist()` 转成二维 Python 列表，`[0]` 取出里面唯一一条向量。这样传给 Milvus 的就是一个 list[float]。
    
    # 库里存纯图片向量，查询时用图文联合向量。 
    # 换一句查询文字，能够改变这次搜索的方向，图库的向量可以继续复用。
    def encode_query(self, image_path: str, text: str) -> list[float]:
        """生成这次搜索用的查询向量"""
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> list[float]:
        """给图库里的候选图片建库"""
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]


def visualize_results(query_image_path: str, retrieved_images: list, img_height: int = 300, img_width: int = 300, row_count: int = 4) -> np.ndarray:
    """从检索到的图像列表创建一个全景图用于可视化。"""
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # 处理查询图像
    query_pil = Image.open(query_image_path).convert("RGB")
    query_cv = np.array(query_pil)[:, :, ::-1]
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    query_display_area[img_height * (row_count - 1):, :] = cv2.resize(bordered_query, (img_width, img_height))
    cv2.putText(query_display_area, "Query", (10, panoramic_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 处理检索到的图像
    for i, img_path in enumerate(retrieved_images):
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width
        
        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved
        
        # 添加索引号
        cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return np.hstack([query_display_area, panoramic_image])

# 3. 初始化客户端
print("--> 正在初始化编码器和 Milvus 客户端...")
encoder = Encoder(MODEL_NAME, MODEL_PATH)
milvus_client = MilvusClient(uri=MILVUS_URI)    # 连接 Docker 启动的数据库

# 4. 创建 Milvus Collection
print(f"\n--> 正在创建 Collection '{COLLECTION_NAME}'")
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"已删除已存在的 Collection: '{COLLECTION_NAME}'")

# 扫描 data/C3/dragon 下的 PNG
image_list = glob(os.path.join(DATA_DIR, "dragon", "*.png"))
if not image_list:
    raise FileNotFoundError(f"在 {DATA_DIR}/dragon/ 中未找到任何 .png 图像。")
print(image_list)

# 尝试在建库的时候，把 query.png 排除，不然查询出来自己肯定排第一
# image_list = [image for image in image_list if image != '../../data/C3/dragon/query.png']

# Collection 可以理解成数据库里的表。
# 代码先用第一张图的编码结果算出 dim，再用 CollectionSchema 规定这张表的三个字段
dim = len(encoder.encode_image(image_list[0]))

fields = [
    # 主键字段，设置自增 (auto_id=True)
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    # 向量字段，维度与模型的输出向量维度一致
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    # 存储原图像路径的标量字段，最大长度限制 512 字符
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
]

# 创建集合 Schema
schema = CollectionSchema(fields, description="多模态图文检索")
print("Schema 结构:")
print(schema)

# 创建集合
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
print(f"成功创建 Collection: '{COLLECTION_NAME}'")
print("Collection 结构:")
print(milvus_client.describe_collection(collection_name=COLLECTION_NAME))

# 5. 准备并插入数据
# 随后，脚本遍历图片，用 encode_image() 生成向量，
# 攒成一条条 {"vector": vector, "image_path": image_path}，再调用 insert() 一次写入。
# 图片文件仍在本地目录，Milvus 里存的是向量和路径。
print(f"\n--> 正在向 '{COLLECTION_NAME}' 插入数据")
data_to_insert = []
for image_path in tqdm(image_list, desc="生成图像嵌入"):
    vector = encoder.encode_image(image_path)
    data_to_insert.append({"vector": vector, "image_path": image_path})

if data_to_insert:
    result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"成功插入 {result['insert_count']} 条数据。")

# 6. 创建索引
# HNSW 可以先理解成一张向量之间的近邻地图。搜索时，Milvus 沿着图里的连接寻找相似向量。
# 这种近似搜索在图库变大时能减少需要比较的向量数。
print(f"\n--> 正在为 '{COLLECTION_NAME}' 创建索引")
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    # M=16：控制建图时每个向量能连接多少近邻
    # efConstruction=256：控制建图时考虑的候选范围
    params={"M": 16, "efConstruction": 256}
)
milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
print("成功为向量字段创建 HNSW 索引。")
print("索引详情:")
print(milvus_client.describe_index(collection_name=COLLECTION_NAME, index_name="vector"))
milvus_client.load_collection(collection_name=COLLECTION_NAME)
print("已加载 Collection 到内存中。")

# 7. 执行多模态检索
print(f"\n--> 正在 '{COLLECTION_NAME}' 中执行检索")
query_image_path = os.path.join(DATA_DIR, "dragon", "query.png")
query_text = "一条龙"
query_vector = encoder.encode_query(image_path=query_image_path, text=query_text)

search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    # data=[query_vector] 多包了一层列表，因为 Milvus 支持一次提交多条查询
    data=[query_vector],
    # output_fields=["image_path"] 让命中结果带回图片路径
    output_fields=["image_path"],
    limit=6,
    # ef=128：控制搜索探索的范围。
    search_params={"metric_type": "COSINE", "params": {"ef": 128}}
)[0]

retrieved_images = []
print("检索结果:")
for i, hit in enumerate(search_results):
    print(f"  Top {i+1}: ID={hit['id']}, 距离={hit['distance']:.4f}, 路径='{hit['entity']['image_path']}'")
    retrieved_images.append(hit['entity']['image_path'])

# 8. 可视化与清理
print(f"\n--> 正在可视化结果并清理资源")
if not retrieved_images:
    print("没有检索到任何图像。")
else:
    # 用 Pillow 读取图片，用 OpenCV 缩放、加边框和编号，再用 NumPy 把左侧查询图和右侧结果网格拼在一起。
    # 拼好的图保存到 data/C3/search_result.png。
    panoramic_image = visualize_results(query_image_path, retrieved_images)
    combined_image_path = os.path.join(DATA_DIR, "search_result.png")
    cv2.imwrite(combined_image_path, panoramic_image)
    print(f"结果图像已保存到: {combined_image_path}")
    Image.open(combined_image_path).show()

milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")


# Milvus 的流程可以记成：建表 → 写入 → 建索引 → 加载 → 搜索。前提是先用 MilvusClient 连上服务。
#
# 1. 建表：`create_collection()`
# 先定义 Schema，告诉 Milvus 每条记录有哪些字段。这里是 id、768 维的 vector 和原图路径 image_path。
# 2. 写入：`insert()`
# Python 用 BGE 把图库里的图片转成向量，再把「向量 + 图片路径」一条条写入集合。`id` 由 Milvus 自动生成。
# 3. 建索引：`create_index()`
# 给 vector 字段建立 HNSW 索引，并指定用 COSINE 比较向量。这一步决定 Milvus 怎样找相似向量。
# 4. 加载：`load_collection()`
# 让 Milvus 把索引和查询需要的数据加载到服务端内存，准备接收查询。
# 5. 搜索：`search()`
# Python 把「查询图片 + 一条龙」转成查询向量，交给 Milvus 找 Top 5。Milvus 返回相似度和 image_path，Python 再按路径打开原图。
# 6. 用完后按需清理
# release_collection() 释放查询内存，保留数据和索引，下次加载后还能查。`drop_collection()` 会删除整个集合，教程用它来清理演示数据。
#
# 实际使用时，建表、写入和建索引属于建库阶段。集合加载好以后，用户每问一次，只需要生成新的查询向量 → search()，不用把图库重新编码一遍。

