# 文件用途：使用Word2Vec和GloVe计算文本相似度
# 对应章节/小节：第1章 1.1.5 代码实战
# 作者：NYY
# 创建日期：2024-10-22（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# NLP之TextSimil：基于两份文档先分词再利用词嵌入模型(Word2Vec/GloVe)实现词语向量表示然后平均向量最后利用余弦相似度法来计算文本相似度实现代码
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Step1、准备文档
doc1 = "This document is "
doc2 = "This document "

# Step2、将文档分词
# 文本分词
tokens1 = word_tokenize(doc1.lower())
tokens2 = word_tokenize(doc2.lower())
print('tokens1: ', tokens1)

# Step3、分别构造 Word2Vec 和 GloVe 文档向量并比较相似度
# 本脚本在词嵌入表示中比较两种词向量。
# T1、Word2Vec：训练词向量模型
# 分词后的结果作为输入数据来训练Word2Vec模型。模型将学习每个词语的词向量表示
sentences = [tokens1, tokens2]
# 模型的训练过程是通过迭代语料库中的句子来进行的。对于每个句子，Word2Vec模型会根据选定的训练算法（CBOW或Skip-gram）来更新词向量，并不断调整它们的表示，以提高对上下文关系的捕捉能力。
# 两句语料仅用于演示流程，训练出的词向量不具备可靠的语义质量。
model = Word2Vec(sentences, min_count=1, sg=0, seed=42, workers=1)  # sg=0即默认使用CBOW，min_count=1表示只考虑出现次数≥1的词语，即保留所有词语。
print(model)
word2vec01 = [model.wv[word] for word in tokens1 if word in model.wv]
word2vec02 = [model.wv[word] for word in tokens2 if word in model.wv]
print(tokens1[0], model.wv[tokens1[0]].shape, list(model.wv[tokens1[0]]))

# 计算文档的平均向量：将所有词向量加起来然后除以词向量的数量，得到一个单一的向量(长度为词嵌入维度)，这个向量可以被视为整个文档的向量表示。
doc_vec1 = np.mean(word2vec01, axis=0)
doc_vec2 = np.mean(word2vec02, axis=0)
print('doc_vec1',doc_vec1.shape)
print('doc_vec2',doc_vec2.shape)

# 计算余弦相似度
similarity = cosine_similarity([doc_vec1], [doc_vec2])[0][0]
print("Word2Vec Similarity:", similarity)

# T2、GloVe：载入词向量模型
import nltk
from gensim.models import KeyedVectors

# 访问GloVe官方网站：https://nlp.stanford.edu/projects/glove/
# 在网页上找到适合你需求的预训练词向量文件（如GloVe.6B.zip）并下载
# 下载并加载预训练的词向量模型
# nltk.download('punkt')
path_to_glove = 'glove.6B.100d.txt'
glove = KeyedVectors.load_word2vec_format(path_to_glove, binary=False, no_header=True)

# 全部词都不在词表时，平均值没有定义，直接给出明确提示。
def mean_vector(tokens, vectors):
    available = [vectors[word] for word in tokens if word in vectors]
    if not available:
        raise ValueError("文档中没有词出现在 GloVe 词表中")
    return np.mean(available, axis=0)

doc_vec1 = mean_vector(tokens1, glove)
doc_vec2 = mean_vector(tokens2, glove)

# 计算余弦相似度
similarity = cosine_similarity([doc_vec1], [doc_vec2])[0][0]
print("Glove Similarity:", similarity)
