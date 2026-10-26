# 文件用途：演示自动提示工程
# 对应章节/小节：第7章 7.3.5 提示设计方法论分类
# 作者：NYY
# 创建日期：2024-03-29（文件系统记录）
# 运行方式：直接运行以查看模拟输出
# 备注：使用模拟响应展示提示流程

"""用模拟响应说明候选提示生成、评分与精炼。"""

from typing import List


def llm_response(question: str) -> str:
    """返回示例文本；需要真实调用时替换此函数。"""
    return f"请清晰、具体地回答：{question}"


# Step1、生成不同表述的候选提示
def generate_prompts(task_description: str, num_prompts: int = 5) -> List[str]:
    if num_prompts < 1:
        raise ValueError("num_prompts 必须大于零")
    approaches = ["用简短语言", "给出具体步骤", "用日常例子", "先定义再解释", "按要点"]
    return [
        llm_response(f"{approaches[index % len(approaches)]}说明{task_description}")
        for index in range(num_prompts)
    ]


# Step2、按给定标准为提示评分并精炼
def score_prompt(prompt: str, criteria: List[str]) -> float:
    if not criteria:
        raise ValueError("criteria 不能为空")
    # 仅按提示是否含有关键词评分，不是模型回答质量或任务正确率。
    return sum(criterion in prompt for criterion in criteria) / len(criteria)


def refine_prompt(prompt: str, feedback: str) -> str:
    return f"{prompt} {feedback}"


# Step3、选出示例提示并展示精炼结果
def main() -> None:
    task_description = "描述一个复杂的科学概念，使其易于理解。"
    prompts = generate_prompts(task_description)
    criteria = ["清晰", "具体", "易于理解"]
    best_prompt = max(prompts, key=lambda prompt: score_prompt(prompt, criteria))
    refined_prompt = refine_prompt(best_prompt, "增加一个日常生活中的例子。")
    print("候选提示:", prompts)
    print("启发式规则选出的提示:", best_prompt)
    print("精炼后的提示:", refined_prompt)


if __name__ == "__main__":
    main()
