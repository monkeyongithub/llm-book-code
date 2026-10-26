# 文件用途：使用句向量模型计算句子相似度
# 对应章节/小节：第1章 1.1.5 代码实战
# 作者：NYY
# 创建日期：2024-10-22（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# NLP之TextSimil：基于两个句子利用预训练的句子嵌入模型(sbert-base-chinese-nli)转换为高维向量表示再利用余弦距离法来计算相似度实现代码
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances

# Step1、加载预训练中文句向量模型
model = SentenceTransformer("uer/sbert-base-chinese-nli")
# Step2、编码待比较的句子
sentences = ["那个人很开心", "那个人非常开心"]
sentence_embeddings = model.encode(sentences)
# Step3、将余弦距离换算为相似度
cosine_score = 1 - paired_cosine_distances(
    [sentence_embeddings[0]], [sentence_embeddings[1]]
)[0]
print(f"句子相似度：{cosine_score:.4f}")
