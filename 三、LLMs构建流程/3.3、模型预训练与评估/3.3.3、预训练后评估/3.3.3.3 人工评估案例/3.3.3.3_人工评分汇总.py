# 文件用途：按书稿量表汇总人工评分并排除未评价标记
# 对应章节/小节：第3章 3.3.3.3 代码实战：人工评估案例
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从代码根目录运行本文件的相对路径
# 备注：配合书稿概念阅读的小型演示

"""分数为相对参考答案的 -2～2 级评价；-100 表示尚未评价。"""

from collections import Counter
from statistics import mean


# Step1、准备独立评分记录；这里的分数仅为演示数据
ratings = [
    {"sample": "样本1", "rater": "甲", "score": 2},
    {"sample": "样本1", "rater": "乙", "score": 1},
    {"sample": "样本2", "rater": "甲", "score": -1},
    {"sample": "样本2", "rater": "乙", "score": -100},
]


# Step2、校验量表范围，排除尚未评价的记录
def summarize_ratings(records):
    valid, pending = [], 0
    seen = set()
    for record in records:
        key = (record["sample"], record["rater"])
        if key in seen:
            raise ValueError("同一评分者对同一样本有重复记录")
        seen.add(key)
        score = record["score"]
        if type(score) is not int or score not in (-100, -2, -1, 0, 1, 2):
            raise ValueError("评分不符合书稿量表")
        if score == -100:
            pending += 1
        else:
            valid.append(score)
    return {"已评价": len(valid), "未评价": pending,
            "评分记录均值": mean(valid) if valid else None,
            "分数分布": dict(sorted(Counter(valid).items()))}


# Step3、同时展示均值、覆盖数量与分布，避免把未评价当作低分
if __name__ == "__main__":
    print(summarize_ratings(ratings))
    # 本例按评分记录等权平均；比较模型时应保持样本与评分人数一致。
    # 量表为等级评价，均值仅作汇总，仍需检查分歧及具体回答。
