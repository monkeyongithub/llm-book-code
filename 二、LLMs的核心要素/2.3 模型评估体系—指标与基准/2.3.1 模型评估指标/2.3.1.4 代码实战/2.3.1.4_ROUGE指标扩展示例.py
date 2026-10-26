# 文件用途：演示ROUGE文本生成评估
# 对应章节/小节：第2章 2.3.1.4 代码实战
# 作者：NYY
# 创建日期：2024-10-23（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""按空格分词演示 ROUGE；中文文本需先分词。所有结果取值为 0～1。"""

from collections import Counter


def overlap_scores(overlap, reference_count, candidate_count):
    """本例约定：分母为零时对应指标为零。"""
    recall = overlap / reference_count if reference_count else 0.0
    precision = overlap / candidate_count if candidate_count else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return recall, precision, f1


# T1、ROUGE-N：比较连续 n 元组，重复次数以参考文本为上限。
# Step1、构造 n 元组并计算裁剪后的重叠数
def calculate_rouge_n(reference, candidate, n=1):
    if not isinstance(n, int) or n < 1:
        raise ValueError("n 必须为正整数")
    def ngrams(text):
        tokens = text.split()
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    reference_ngrams, candidate_ngrams = ngrams(reference), ngrams(candidate)
    overlap = sum((reference_ngrams & candidate_ngrams).values())
    return overlap_scores(overlap, sum(reference_ngrams.values()), sum(candidate_ngrams.values()))


def calculate_rouge_1(reference, candidate):
    return calculate_rouge_n(reference, candidate, n=1)


def calculate_rouge_2(reference, candidate):
    return calculate_rouge_n(reference, candidate, n=2)


# T2、ROUGE-L：用最长公共子序列保留顺序信息。
# Step2、通过动态规划计算最长公共子序列长度
def lcs_length(reference_tokens, candidate_tokens):
    previous = [0] * (len(candidate_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for j, candidate_token in enumerate(candidate_tokens, 1):
            current.append(previous[j - 1] + 1 if reference_token == candidate_token
                           else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def calculate_rouge_l(reference, candidate):
    reference_tokens, candidate_tokens = reference.split(), candidate.split()
    overlap = lcs_length(reference_tokens, candidate_tokens)
    return overlap_scores(overlap, len(reference_tokens), len(candidate_tokens))


# Step3、比较重复词、词序变化及空文本对各指标的影响
if __name__ == "__main__":
    reference = "it is cold outside"
    candidates = ["it is not cold outside", "cold cold cold cold", "outside cold it is", ""]
    for candidate in candidates:
        print(f"候选：{candidate!r}")
        for name, metric in [("ROUGE-1", calculate_rouge_1), ("ROUGE-2", calculate_rouge_2),
                             ("ROUGE-L", calculate_rouge_l)]:
            recall, precision, f1 = metric(reference, candidate)
            print(f"{name}: R={recall:.3f}, P={precision:.3f}, F1={f1:.3f}")
    # 重复词裁剪是 ROUGE-N 的计数规则，并非独立算法；词面重叠也不保证事实正确。
