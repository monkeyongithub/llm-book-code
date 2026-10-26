# 文件用途：比较二元语言模型的最大似然估计与加一平滑
# 对应章节/小节：第1章 1.1.1 SLM(统计语言模型)
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从代码根目录运行本文件的相对路径
# 备注：配合书稿概念阅读的小型演示

"""用已分词的小语料观察 P(下一个词 | 前一个词)。"""

from collections import Counter
from math import exp, log


# Step1、统计二元组和上下文次数，保留开始、结束与未知词标记
corpus = [["我", "喜欢", "读书"], ["我", "喜欢", "编程"], ["你", "喜欢", "读书"]]
vocabulary = {word for sentence in corpus for word in sentence} | {"</s>", "<unk>"}
bigrams, contexts = Counter(), Counter()
for sentence in corpus:
    tokens = ["<s>", *sentence, "</s>"]
    for previous, current in zip(tokens, tokens[1:]):
        bigrams[previous, current] += 1
        contexts[previous] += 1


# Step2、用同一组计数比较两种概率估计
def probability(previous, current, alpha=0.0):
    if alpha < 0:
        raise ValueError("平滑系数不能为负")
    previous = previous if previous in vocabulary or previous == "<s>" else "<unk>"
    current = current if current in vocabulary else "<unk>"
    denominator = contexts[previous] + alpha * len(vocabulary)
    return (bigrams[previous, current] + alpha) / denominator if denominator else 0.0


# Step3、在对数空间累加句子概率，展示未见二元组的影响
def sentence_probability(words, alpha):
    tokens = ["<s>", *words, "</s>"]
    values = [probability(a, b, alpha) for a, b in zip(tokens, tokens[1:])]
    return 0.0 if any(value == 0 for value in values) else exp(sum(log(value) for value in values))


if __name__ == "__main__":
    for label, alpha in [("T1、最大似然估计", 0.0), ("T2、加一平滑", 1.0)]:
        print(label)
        print("给定‘喜欢’的下一词分布：", {word: round(probability("喜欢", word, alpha), 3)
                                         for word in sorted(vocabulary)})
        print("未出现过的完整句子‘你 喜欢 编程’：", sentence_probability(["你", "喜欢", "编程"], alpha))
        print("含未知词的句子：", sentence_probability(["我", "喜欢", "音乐"], alpha))
    # 平滑为未见事件保留概率，不意味着模型已经理解未知词的语义。
