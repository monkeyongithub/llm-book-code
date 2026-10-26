# 文件用途：演示自洽性
# 对应章节/小节：第7章 7.3.5 提示设计方法论分类
# 作者：NYY
# 创建日期：2024-03-29（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：使用模拟回答说明多数投票

from collections import Counter


# Step1、生成同一问题的多次示例回答
def llm_responses(question):
    """返回用于说明多数投票的模拟回答。"""
    return ["18", "18", "20"]


# Step2、统计出现频率最高的回答
def select_consistent_answer(responses):
    """按多数投票选择答案，并计算其占比。"""
    if not responses:
        raise ValueError("候选回答不能为空")
    counts = Counter(responses).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        raise ValueError("最高票并列，需增加采样或明确决策规则")
    answer, votes = counts[0]
    return answer, votes / len(responses)


# Step3、输入问题并展示多数回答
if __name__ == "__main__":
    question = "买 3 本每本 6 元的书，一共多少钱？"
    answer, share = select_consistent_answer(llm_responses(question))
    print(f"多数回答：{answer}（占比：{share:.0%}）")
