# 文件用途：用同一分类任务比较零样本与少样本提示结构
# 对应章节/小节：第7章 7.3.1 ICL上下文学习
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从代码根目录运行本文件的相对路径
# 备注：配合书稿概念阅读的小型演示

"""只构造提示，展示上下文示例的格式与顺序；不进行参数训练。"""


# Step1、准备统一任务和与待预测文本不同的已标注示例
instruction = "判断英文句子的情感，只输出 Positive 或 Negative。"
examples = [("The weather is sunny.", "Positive"),
            ("It's raining cats and dogs.", "Negative"),
            ("I feel very happy today.", "Positive")]
query = "The movie was fantastic!"
# 示例标签沿用书稿的任务设定；天气描述的情感可能依语境变化，应在真实任务中明确标准。


# Step2、按输入—输出模板组织演示，并把新问题放在末尾
def build_prompt(question, demonstrations):
    parts = [instruction]
    for text, label in demonstrations:
        parts.append(f"Text: {text}\nLabel: {label}")
    parts.append(f"Text: {question}\nLabel:")
    return "\n\n".join(parts)


# Step3、比较示例数量和顺序对提示结构的影响
if __name__ == "__main__":
    # T1、零样本：仅有任务指令。
    print("T1 零样本：\n", build_prompt(query, []))
    # T2、少样本：加入书稿列出的三个演示。
    print("T2 少样本：\n", build_prompt(query, examples))
    print("调整示例顺序：\n", build_prompt(query, list(reversed(examples))))
    # 是否改善模型回答需要在同一验证集上评估，不能由提示长度直接判断。
