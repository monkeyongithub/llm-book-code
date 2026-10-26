# 文件用途：区分词元 ID、填充掩码、训练标签与嵌入向量
# 对应章节/小节：第3章 3.2.4.1 构建输入特征
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从代码根目录运行本文件的相对路径
# 备注：配合书稿概念阅读的小型演示

"""用固定小词表演示输入特征；嵌入矩阵是示例数值，不具有训练后的语义。"""

import numpy as np


# Step1、将已分词文本映射到固定词表中的整数 ID
vocabulary = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3,
              "我": 4, "喜欢": 5, "读书": 6, "你": 7}
sentences = [["我", "喜欢", "读书"], ["你", "读书"]]
sequences = [[vocabulary["[BOS]"], *[vocabulary.get(word, 1) for word in words],
              vocabulary["[EOS]"]] for words in sentences]

# Step2、填充到等长，分别建立注意力掩码和忽略填充位置的标签
width = max(map(len, sequences))
input_ids = np.array([ids + [0] * (width - len(ids)) for ids in sequences])
attention_mask = (input_ids != vocabulary["[PAD]"]).astype(int)
labels = np.where(attention_mask == 1, input_ids, -100)

# Step3、在模型内部按 ID 查表；预处理输出的整数不是语义向量
embedding_table = np.random.default_rng(42).normal(size=(len(vocabulary), 4))
embeddings = embedding_table[input_ids]

# Step4、观察张量形状与下一词预测目标
if __name__ == "__main__":
    print("input_ids：\n", input_ids)
    print("attention_mask：\n", attention_mask)
    print("labels：\n", labels)
    print("嵌入形状 [batch, sequence, hidden]：", embeddings.shape)
    print("下一词目标（用于说明内部移位）：\n", labels[:, 1:])
    # 常见 Hugging Face 因果语言模型内部将 logits[:, :-1] 与 labels[:, 1:] 对齐。
    # 传给这类模型的 labels 保持与 input_ids 同长度，不要再次手动移位。
    # attention_mask 用于遮住填充；未来位置还需由模型内部的因果遮罩控制。
