import torch
from visual_bge.visual_bge.modeling import Visualized_BGE


# 加载模型，进入推理模式
model = Visualized_BGE(
    model_name_bge="BAAI/bge-base-en-v1.5",
    model_weight="../../models/bge/Visualized_base_en_v1.5.pth"
)

# model.eval() 把模型切到评估模式，关闭 Dropout 等训练行为。
# 后面的 with torch.no_grad(): 则告诉 PyTorch 这次只做推理，可以省掉记录梯度所需的内存。
model.eval()

# `encode(image=..., text=...)` 会先把图片变成一组视觉特征，再把这些特征和文字一起送进 BGE 编码，最后输出一个代表图文组合的向量。
with torch.no_grad():
    # text_emb = model.encode(text="datawhale开源组织的logo")
    # img_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png")
    # multi_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png", text="datawhale开源组织的logo")
    # img_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png")
    # multi_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png", text="datawhale开源组织的logo")

    text_emb = model.encode(text="blue whale")
    img_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png")
    multi_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png", text="blue whale")
    img_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png")
    multi_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png", text="blue whale")

# 计算相似度
# 模型默认把最终向量归一化为单位长度，所以这里的点积就是余弦相似度。值越高，两个向量的方向越接近。
sim_1 = img_emb_1 @ img_emb_2.T         
sim_2 = img_emb_1 @ multi_emb_1.T       
sim_3 = text_emb @ multi_emb_1.T        
sim_4 = multi_emb_1 @ multi_emb_2.T     

print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")         # tensor([[0.8318]])
print(f"图文结合1 vs 纯图像: {sim_2}")      # tensor([[0.8291]])
print(f"图文结合1 vs 纯文本: {sim_3}")      # tensor([[0.7627]])
print(f"图文结合1 vs 图文结合2: {sim_4}")   # tensor([[0.9058]])

# 向量信息分析
print("\n=== 嵌入向量信息 ===")
print(f"多模态向量维度: {multi_emb_1.shape}")   # torch.Size([1, 768])
print(f"图像向量维度: {img_emb_1.shape}")       # torch.Size([1, 768])
print(f"多模态向量示例 (前10个元素): {multi_emb_1[0][:10]}")
print(f"图像向量示例 (前10个元素):   {img_emb_1[0][:10]}")
