# 文件用途：使用词频和TF-IDF计算文本相似度
# 对应章节/小节：第1章 1.1.5 代码实战
# 作者：NYY
# 创建日期：2024-10-22（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# NLP之TextSimil：基于两份文档依次利用词频向量/TF-IDF向量表示并结合余弦相似度实现文本相似度计算实现代码

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 计算两段文本的相似度
# CountVectorizer：只关注词频
# TfidfVectorizer：通过乘以IDF来加权词频，同时考虑了词频和词的重要性

# Step1、准备两份待比较文档
doc1 = "This document is "
doc2 = "This document "
# 简短文本示例：doc2 = "a b c"
# 英文输入示例：doc1 = "This is the first document."
# 英文输入示例：doc2 = "This document is the second document."

# Step2、分别生成词频向量与 TF-IDF 向量
# 本脚本在统计表示中比较两种向量化方法。
# T1、词频表示：使用 CountVectorizer将文档转换为词频向量表示
count_vectorizer = CountVectorizer()
count_vectorizer.fit([doc1, doc2])
count_vector = count_vectorizer.transform([doc1, doc2])
print('count_vector\n', count_vector)
# 遍历每个文档的词频向量并转为普通矩阵
for i in range(count_vector.shape[0]):
    # 获取当前文档的词频向量
    doc_vector = count_vector[i]
    # 将稀疏矩阵转换为普通矩阵并打印
    print(doc_vector.toarray())




# T2、TF-IDF 表示：使用 TfidfVectorizer将文档转换为TF-IDF向量表示
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit([doc1, doc2])
tfidf_vector = tfidf_vectorizer.transform([doc1, doc2])
print('tfidf_vector\n', tfidf_vector)




# Step3、计算并比较两种表示的余弦相似度
# T1、计算词频向量的余弦相似度
count_cosine_sim = cosine_similarity(count_vector)[0, 1]
print("count_vectorizer Similarity", count_cosine_sim)

# T2、计算 TF-IDF 向量的余弦相似度
tfidf_cosine_sim = cosine_similarity(tfidf_vector)[0, 1]
print("tfidf_vectorizer Similarity:", tfidf_cosine_sim)
