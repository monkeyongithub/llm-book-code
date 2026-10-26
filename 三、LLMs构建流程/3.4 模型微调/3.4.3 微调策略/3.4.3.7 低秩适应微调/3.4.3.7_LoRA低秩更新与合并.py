# 文件用途：用小矩阵展示 LoRA 零初始化、低秩更新与推理合并
# 对应章节/小节：第3章 3.4.3.7 低秩适应微调
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从代码根目录运行本文件的相对路径
# 备注：配合书稿概念阅读的小型演示

"""演示 ΔW = (alpha/r) BA；分解的是权重更新，不要求原权重本身低秩。"""

import numpy as np


# Step1、设置固定基础权重与低秩参数；B 初始为零时输出保持不变
rng = np.random.default_rng(42)
in_features, out_features, rank = 6, 4, 2
alpha = 4
x = rng.normal(size=(3, in_features))
base_weight = rng.normal(size=(out_features, in_features))
a = rng.normal(scale=0.01, size=(rank, in_features))
b = np.zeros((out_features, rank))
base_output = x @ base_weight.T
initial_output = base_output + (alpha / rank) * (x @ a.T @ b.T)

# Step2、用非零 B 模拟训练后的适配器，基础权重仍保持不变
# 本例不运行梯度训练，只演示适配器前向计算与合并的代数关系。
b = rng.normal(scale=0.01, size=b.shape)
adapter_output = base_output + (alpha / rank) * (x @ a.T @ b.T)

# Step3、将低秩更新合并到基础权重，比较两条推理路径
merged_weight = base_weight + (alpha / rank) * (b @ a)
merged_output = x @ merged_weight.T

if __name__ == "__main__":
    print("初始新增路径为零：", np.allclose(initial_output, base_output))
    print("合并前后输出一致：", np.allclose(adapter_output, merged_output))
    print("最大绝对差：", np.max(np.abs(adapter_output - merged_output)))
    print("基础权重参数数：", base_weight.size, "适配器参数数：", a.size + b.size)
    # 合并等价性针对关闭 dropout 的线性前向；浮点运算可能存在舍入差异。
