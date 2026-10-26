# 文件用途：演示4位量化与反量化
# 对应章节/小节：第3章 3.5.3.1 模型量化技术
# 作者：NYY
# 创建日期：2024-04-04（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

#######


import numpy as np

# Step1、将浮点数映射到 4 位整数
def quantize_to_4bit(array):
    array = np.asarray(array, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("输入必须是非空的有限数值数组")
    # 找到最大值和最小值
    max_val = np.max(array)
    min_val = np.min(array)
    # 常量数组没有量化区间，直接映射到零。
    if max_val == min_val:
        return np.zeros_like(array, dtype=np.uint8), min_val, 0.0
    step = (max_val - min_val) / 15
    # 将数组按步长量化到4位整数
    quantized_array = np.clip(np.round((array - min_val) / step), 0, 15).astype(np.uint8)
    return quantized_array, min_val, step

# Step2、按原区间恢复近似浮点数
def dequantize_from_4bit(quantized_array, min_val, step):
    # 反量化
    dequantized_array = quantized_array * step + min_val
    return dequantized_array


# 创建一个示例数组
# Step3、用示例数组比较量化前后结果
# 用 uint8 保存 0～15 的量化值；本例没有将两个 4 位值打包进一个字节。
original_array = np.random.default_rng(42).random(10) * 100
# 进行4-bit量化
quantized_array, min_val, step = quantize_to_4bit(original_array)
# 反量化
dequantized_array = dequantize_from_4bit(quantized_array, min_val, step)
# 打印结果
print("原始数组：", original_array)
print("4-bit量化后的数组：", quantized_array)
print("反量化后的数组：", dequantized_array)
