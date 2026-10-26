# 文件用途：演示思维链提示
# 对应章节/小节：第7章 7.3.2 CoT思维链提示
# 作者：NYY
# 创建日期：2024-03-29（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""比较零样本和手动示例思维链提示；用算术任务展示可核对的解题步骤。"""

import os

from zhipuai import ZhipuAI


# Step1、封装同一模型调用，保证两种提示使用相同接口
def ask_model(client, prompt):
    response = client.chat.completions.create(
        model="glm-4", messages=[{"role": "user", "content": prompt}], temperature=0.2,
    )
    return response.choices[0].message.content or ""


# T1、零样本 CoT：不附示例，要求给出关键计算步骤。
# Step2、直接构造解题请求
def zero_shot_cot(question):
    return f"请解答以下问题，列出关键计算步骤并给出最终答案：\n{question}"


# T2、手动 CoT：示例同时包含问题、计算步骤和答案。
# Step3、在新问题前放入人工编写的解题示例
def manual_cot(question, examples):
    sections = [f"示例问题：{q}\n示例解答：{solution}" for q, solution in examples]
    sections.append(f"请按类似方式解答新问题：{question}")
    return "\n\n".join(sections)


# Step4、比较两种提示的输出，并按题目条件核对结果
if __name__ == "__main__":
    client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])
    question = "小明有 5 个苹果，又买了 3 个，送给朋友 2 个，还剩几个？"
    examples = [("小红有 4 支笔，又买了 2 支，送出 1 支，还剩几支？",
                 "先计算买后数量：4 + 2 = 6；再减去送出的 1 支：6 - 1 = 5。答案：5 支。")]
    for name, prompt in [("T1 零样本", zero_shot_cot(question)),
                         ("T2 手动示例", manual_cot(question, examples))]:
        print(f"\n{name}\n{prompt}\n模型回答：{ask_model(client, prompt)}")
    # 本题可独立核验为 5 + 3 - 2 = 6；有步骤的回答也可能出错。
