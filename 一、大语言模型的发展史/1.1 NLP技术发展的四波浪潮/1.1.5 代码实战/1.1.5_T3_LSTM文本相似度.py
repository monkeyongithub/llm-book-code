# 文件用途：使用LSTM表示文档并计算相似度
# 对应章节/小节：第1章 1.1.5 代码实战
# 作者：NYY
# 创建日期：2024-10-22（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# NLP之TextSimil：基于两份文档先利用分词技术将token文本转换为序列再利用LSTM模型(Embedding层+LSTM层+adam优化)来学习文档表示(语义信息的向量表示)接着利用余弦相似度法来计算文本相似度实现代

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Step1、准备训练示例文档
# 英文输入示例：document1 = "This is the first document."
# 英文输入示例：document2 = "You are my baby."
# document2 = document1
document1 = "This is the first document."
document2 = "This document is the second document."

# Step2、将文档编码并填充为等长序列
# 利用Tokenizer将token文本转换为序列
tokenizer = Tokenizer()
tokenizer.fit_on_texts([document1, document2])
sequences = tokenizer.texts_to_sequences([document1, document2])
print('sequences \n', sequences)

# 获取词汇表最大序列长度并填充序列至相同长度
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length)
print('padded_sequences \n', padded_sequences)

# Step3、训练分类器并提取 LSTM 隐层向量
# 构建了一个简单的RNN模型：嵌入层（Embedding）+LSTM层的RNN模型。
# 嵌入层将词索引映射为密集向量表示，
# LSTM层用于学习文档的表示。
# 构建包含嵌入层与 LSTM 层的分类器
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
inputs = Input(shape=(max_length,))
embedded = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(inputs)
document_representation = LSTM(units=128)(embedded)
predictions = Dense(1, activation="sigmoid")(document_representation)
classifier = Model(inputs=inputs, outputs=predictions)

# 两条文本与人为标签只演示训练和取向量的流程，不能据此判断语义相似度质量。
# LSTM 隐层才是用于比较的文档向量；填充位由 mask_zero 忽略。
classifier.compile(optimizer="adam", loss="binary_crossentropy")
classifier.fit(padded_sequences, np.array([0, 1]), epochs=10, verbose=0)

# 提取训练后的 LSTM 隐层表示。
encoder = Model(inputs=inputs, outputs=document_representation)
document_vectors = encoder.predict(padded_sequences)

# Step4、计算两个文档向量的余弦相似度
similarity = cosine_similarity([document_vectors[0]], [document_vectors[1]])[0][0]
print("Document Similarity:", similarity)
