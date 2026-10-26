# 文件用途：计算二分类指标与真实目标词元的困惑度
# 对应章节/小节：第2章 2.3.1.2 常见的通用指标对比：ACC、P、R、F1、PPL
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从代码根目录运行本文件的相对路径
# 备注：配合书稿概念阅读的小型演示

"""分类指标与困惑度服务于不同任务，不能直接比较大小。"""

import math


# Step1、从真实标签与预测标签构造二分类混淆计数
def classification_metrics(expected, predicted):
    if not expected or len(expected) != len(predicted):
        raise ValueError("标签列表必须非空且等长")
    if any(value not in (0, 1) for value in [*expected, *predicted]):
        raise ValueError("本例仅接受 0、1 二分类标签")
    tp = sum(a == b == 1 for a, b in zip(expected, predicted))
    tn = sum(a == b == 0 for a, b in zip(expected, predicted))
    fp = sum(a == 0 and b == 1 for a, b in zip(expected, predicted))
    fn = sum(a == 1 and b == 0 for a, b in zip(expected, predicted))
    # Step2、按分母含义计算各指标；无正例时约定对应指标为零
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"ACC": (tp + tn) / len(expected), "P": precision, "R": recall, "F1": f1}


# Step3、对真实目标 token 的条件概率计算平均负对数似然与 PPL
def perplexity(target_probabilities):
    probabilities = list(target_probabilities)
    if not probabilities or any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities):
        raise ValueError("至少提供一个 0～1 范围内的有限概率")
    if 0 in probabilities:
        return math.inf
    return math.exp(-sum(math.log(p) for p in probabilities) / len(probabilities))


if __name__ == "__main__":
    print(classification_metrics([1, 0, 1, 1, 0], [1, 1, 1, 0, 0]))
    print("目标 token 概率 [0.5, 0.25, 0.5] 的 PPL：", perplexity([0.5, 0.25, 0.5]))
    # PPL 比较需使用相同评估文本、分词方式与掩码规则；这里不包含填充位置。
