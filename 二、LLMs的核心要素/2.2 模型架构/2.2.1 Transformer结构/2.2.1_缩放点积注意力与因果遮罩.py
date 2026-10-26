# 文件用途：演示单头缩放点积注意力及因果遮罩
# 对应章节/小节：第2章 2.2.1 Transformer结构
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从代码根目录运行本文件的相对路径
# 备注：配合书稿概念阅读的小型演示

"""用小矩阵观察注意力权重；不包含完整 Transformer 层。"""

import numpy as np


def attention(query, key, value, causal=False):
    # Step1、计算 QKᵀ，并按键向量维数缩放
    scores = query @ key.T / np.sqrt(key.shape[-1])
    if causal:
        # Step2、遮住未来位置；对角线保留，保证每行至少一个可见位置
        if query.shape[0] != key.shape[0]:
            raise ValueError("本例因果遮罩要求 Q、K 具有相同序列长度")
        mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)

    # Step3、逐行归一化，利用权重对 V 加权求和
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights, weights @ value


if __name__ == "__main__":
    # 三行分别表示三个 token 的向量；为突出计算流程，直接给定 Q、K、V。
    q = np.array([[1., 0.], [0., 1.], [1., 1.]])
    k = q.copy()
    v = np.array([[1., 2.], [3., 4.], [5., 6.]])
    np.set_printoptions(precision=3, suppress=True)
    # T1、全可见注意力：每个位置都可关注其他位置。
    # T2、因果注意力：当前位置只能关注自己和此前的位置。
    for label, causal in [("T1 全可见", False), ("T2 因果遮罩", True)]:
        weights, output = attention(q, k, v, causal=causal)
        print(label, "\n注意力权重：\n", weights, "\n输出：\n", output)
