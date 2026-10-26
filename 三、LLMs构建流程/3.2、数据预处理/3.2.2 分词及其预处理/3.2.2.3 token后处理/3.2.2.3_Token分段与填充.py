# 文件用途：演示token列表分段与填充
# 对应章节/小节：第3章 3.2.2.3 token后处理
# 作者：NYY
# 创建日期：2024-09-15（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# Step1、定义分段与补齐规则
def split_and_pad(text_list, max_token=10, pad_token="[PAD]"):
    """
    对输入的text_list按照max_token分割，并将最后不足max_token长度的列表填充至max_token长度。

    参数:
    - text_list (list): 输入的文本token列表。
    - max_token (int): 每个分段的最大token数。
    - pad_token (str): 用于填充的特殊token。

    返回:
    - result (list): 分割后的文本列表，每个子列表长度等于max_token。
    """
    if not isinstance(max_token, int) or max_token <= 0:
        raise ValueError("max_token 必须是正整数")
    # 初始化结果列表
    result = []

    # 按照max_token分割文本
    for i in range(0, len(text_list), max_token):
        chunk = text_list[i:i + max_token]

        # 如果最后一段长度不足max_token，填充pad_token
        if len(chunk) < max_token:
            chunk.extend([pad_token] * (max_token - len(chunk)))

        result.append(chunk)

    return result


# 示例用法
# Step2、准备 token 序列并调用分段函数
text_list = ["人工智能", "(AI)", "是一种", "正在", "迅速", "发展", "的", "技术", "。", "越来越多", "的", "领域",
             "正在", "使用", "AI", "技术", "，", "从", "医疗", "到", "金融", "，", "各行各业", "都在", "受益于",
             "它", "的", "强大", "能力", "。"]

max_token = 8  # 每段最多 8 个 token
result = split_and_pad(text_list, max_token=max_token)

# 打印结果
# Step3、查看每段长度与填充结果
for i, chunk in enumerate(result):
    print(len(chunk))
    print(f"Chunk {i + 1}: {chunk}")
