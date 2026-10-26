# 文件用途：演示专家提示
# 对应章节/小节：第7章 7.3.5 提示设计方法论分类
# 作者：NYY
# 创建日期：2024-03-29（文件系统记录）
# 运行方式：直接运行以查看模拟输出
# 备注：使用模拟响应展示提示流程

"""用模拟响应说明如何组合不同领域的专家提示。"""

from typing import List


def llm_response(prompt: str) -> str:
    """返回示例回答；需要真实调用时替换此函数。"""
    return f"示例观点：{prompt}"


# Step1、按不同专家领域构造提示并收集观点
def generate_expert_prompt(question: str, expert_fields: List[str]) -> str:
    if not expert_fields:
        raise ValueError("expert_fields 不能为空")
    viewpoints = []
    for field in expert_fields:
        prompt = f"假设您是一位{field}的专家，请回答：{question}"
        viewpoints.append(llm_response(prompt))
    return "\n".join(viewpoints)


# Step2、汇总专家观点并请求综合回答
def main() -> None:
    question = "气候变化对农业生产有哪些影响？"
    fields = ["农业科学", "环境科学", "经济学", "政治学"]
    viewpoints = generate_expert_prompt(question, fields)
    summary_prompt = f"请参考以下多位专家的观点并作系统总结：\n{viewpoints}"
    print(summary_prompt)
    print(llm_response(summary_prompt))


if __name__ == "__main__":
    main()
