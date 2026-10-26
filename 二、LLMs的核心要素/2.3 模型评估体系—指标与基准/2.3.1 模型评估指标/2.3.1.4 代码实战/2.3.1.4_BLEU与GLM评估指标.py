# 文件用途：演示BLEU等文本生成评估
# 对应章节/小节：第2章 2.3.1.4 代码实战
# 作者：NYY
# 创建日期：2024-10-23（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# (1)、使用nltk库计算文本BLEU分数
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
# 封装计算BLEU分数的函数
# T1、单句 BLEU 评估
# Step1、定义单句 BLEU 计算方法
def calculate_bleu(reference_text, candidate_text):
    # 将参考文本和候选文本都转化为单词列表
    reference = [reference_text.split()]
    candidate = candidate_text.split()

    # 使用平滑函数 method4
    smoothie = SmoothingFunction().method4

    # 计算BLEU分数
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)

    return bleu_score

# 参考文本
# Step2、比较多个候选文本的 BLEU 分数
reference_text = "I am very happy to say that I am drinking a warm cup of tea."

# 生成文本
outputs = [
    "I am very happy that I am drinking a cup of tea.",
    "I am very happy that I am drinking a warm cup of tea.",
    "I am very happy to say that I am drinking a warm tea.",
    "I am very happy to say that I am drinking a warm cup of tea."
]

# 依次计算每个生成文本的BLEU分数
for i, output in enumerate(outputs, 1):
    score = calculate_bleu(reference_text, output)
    print(f"Output {i} BLEU Score: {score:.4f}")





# (2)、GLM-4模型项目中ROUGE和BLEU指标的代码实现
# 源代码地址：https://github.com/THUDM/GLM-4/blob/main/finetune_demo/finetune.py
# T2、GLM 示例中的 ROUGE 与 BLEU 综合评估
# Step3、展示 GLM 评估中的 ROUGE 与 BLEU 汇总函数
def compute_metrics(eval_preds, tokenizer):
    import jieba
    import numpy as np
    from rouge import Rouge

    # 接收生成的 token ID（不是三维 logits）；标签 -100 不属于词表。
    # 与书稿约定一致：ROUGE 返回百分制，BLEU 返回 0～1。
    # 从eval_preds中提取预测和标签的ID列表
    batched_pred_ids, batched_label_ids = eval_preds
    batched_pred_ids = np.asarray(batched_pred_ids)
    batched_label_ids = np.asarray(batched_label_ids)
    if batched_pred_ids.ndim != 2 or batched_label_ids.ndim != 2:
        raise ValueError("请传入二维生成 ID 和标签 ID；训练评估需启用生成预测")
    if len(batched_pred_ids) != len(batched_label_ids):
        raise ValueError("预测和标签的样本数必须一致")
    # 初始化一个字典metrics_dct，用于存储四种评估指标（ROUGE-1、ROUGE-2、ROUGE-L和BLEU-4）的得分列表
    metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
    # for循环遍历预测和标签的ID对，使用tokenizer.decode方法将它们转换为文本，并使用jieba.cut进行中文分词。之后，使用Rouge类计算ROUGE分数，并将ROUGE分数的F值添加到相应的列表中
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        pred_txt = tokenizer.decode([int(i) for i in pred_ids if i != -100], skip_special_tokens=True).strip()
        label_txt = tokenizer.decode([int(i) for i in label_ids if i != -100], skip_special_tokens=True).strip()
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
        if not pred_tokens or not label_tokens:
            for values in metrics_dct.values():
                values.append(0.0)
            continue
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
        for k, v in scores[0].items():
            metrics_dct[k].append(round(v['f'] * 100, 4))
        # 对于BLEU-4分数，函数使用sentence_bleu函数计算预测文本与标签文本之间的BLEU分数，并使用指定的平滑函数method3。计算出的BLEU分数也被添加到metrics_dct字典的相应列表中。
        metrics_dct['bleu-4'].append(
            sentence_bleu([label_tokens], pred_tokens, smoothing_function=SmoothingFunction().method3))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics_dct.items()}
