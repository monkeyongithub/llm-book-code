# 文件用途：比较贪心、温度、top-k 和 top-p 的单步词元选择
# 对应章节/小节：第2章 2.5 模型解码机制
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从代码根目录运行本文件的相对路径
# 备注：配合书稿概念阅读的小型演示

"""给定固定 logits，观察不同解码策略的候选分布；不加载模型。"""

import numpy as np


# Step1、将 logits 转为稳定的温度概率分布
def probabilities(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 1 or not logits.size or not np.isfinite(logits).all():
        raise ValueError("logits 必须是非空的一维有限数值数组")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("温度必须是有限正数；贪心选择直接使用 argmax")
    scores = (logits - logits.max()) / temperature
    weights = np.exp(scores)
    return weights / weights.sum()


# Step2、按候选排名截断，再重新归一化
def truncate(distribution, top_k=None, top_p=None):
    distribution = np.asarray(distribution, dtype=float)
    if (distribution.ndim != 1 or not distribution.size or not np.isfinite(distribution).all()
            or (distribution < 0).any() or not np.isclose(distribution.sum(), 1)):
        raise ValueError("输入必须为非负且总和为 1 的概率分布")
    order = np.argsort(-distribution, kind="stable")
    if top_k is not None:
        if not isinstance(top_k, int) or not 1 <= top_k <= len(order):
            raise ValueError("top_k 必须在 1 到词表大小之间")
        order = order[:top_k]
    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p 必须位于 (0, 1]")
        cumulative = np.cumsum(distribution[order] / distribution[order].sum())
        # 保留第一个使累计概率达到阈值的 token，不能把越过阈值的那个删掉。
        count = min(np.searchsorted(cumulative, top_p, side="left") + 1, len(order))
        order = order[:count]
    result = np.zeros_like(distribution)
    result[order] = distribution[order]
    return result / result.sum()


# Step3、对同一组 logits 比较选择结果与候选概率
if __name__ == "__main__":
    tokens = np.array(["读书", "编程", "散步", "旅行", "睡觉"])
    logits = np.array([3.0, 2.5, 1.4, 0.3, -0.5])
    rng = np.random.default_rng(42)
    # T1、贪心：选择 logits 最大的词元。
    print("T1 贪心：", tokens[logits.argmax()])
    # T2、温度采样；T3、top-k；T4、top-p。
    methods = [("T2 温度 0.7", probabilities(logits, 0.7)),
               ("T3 top-k=2", truncate(probabilities(logits), top_k=2)),
               ("T4 top-p=0.8", truncate(probabilities(logits), top_p=0.8))]
    for label, distribution in methods:
        chosen = rng.choice(len(tokens), p=distribution)
        print(label, dict(zip(tokens, distribution.round(3))), "采样：", tokens[chosen])
    # 单步分布只能说明选择机制，不能据此评判整段文本质量。
