# 文件用途：比较文本标准化后的精确去重与字符二元组近似检测
# 对应章节/小节：第3章 3.1.2.2 去重过滤
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从代码根目录运行本文件的相对路径
# 备注：配合书稿概念阅读的小型演示

"""只输出保留文本及疑似重复对，不读取或删除原始语料文件。"""

import unicodedata


# Step1、统一全半角、大小写与连续空白
def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


# T1、精确去重：规范化文本相同则保留第一次出现的原文。
def exact_deduplicate(documents):
    seen, kept = set(), []
    for text in documents:
        key = normalize(text)
        if key and key not in seen:
            seen.add(key)
            kept.append(text)
    return kept


# T2、近似检测：按字符二元组集合计算 Jaccard 相似度。
# Step2、比较内容重叠程度，将超过阈值的文本对交给人工检查
def shingles(text):
    text = normalize(text)
    return {text[i:i + 2] for i in range(len(text) - 1)} if len(text) >= 2 else ({text} if text else set())


def near_duplicates(documents, threshold=0.8):
    if not 0 < threshold <= 1:
        raise ValueError("阈值必须位于 (0, 1]")
    features = [shingles(text) for text in documents]
    pairs = []
    for i, left in enumerate(features):
        for j in range(i + 1, len(features)):
            union = left | features[j]
            similarity = len(left & features[j]) / len(union) if union else 0.0
            if similarity >= threshold:
                pairs.append((i, j, similarity))
    return pairs


# Step3、观察精确重复与相似文本，理解近似去重存在误删风险
if __name__ == "__main__":
    texts = ["模型需要高质量训练数据。", "模型需要高质量训练数据。",
             "模型需要高质量训练数据！", "模型不需要高质量训练数据。"]
    kept = exact_deduplicate(texts)
    print("精确去重后：", kept)
    for i, j, score in near_duplicates(kept):
        print(f"疑似重复（{score:.2f}）：{kept[i]!r} / {kept[j]!r}")
    # 否定词改变含义却可能保持很高的字符重叠；不能只按相似度自动删除。
    # 本例两两比较为 O(n²)，用于理解概念；大语料可参考书稿中的 MinHash/LSH。
