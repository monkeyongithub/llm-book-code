# 文件用途：演示提示约束
# 对应章节/小节：第7章 7.3.5 提示设计方法论分类
# 作者：NYY
# 创建日期：2024-03-29（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

#######

'''
为了实现这个策略，我们将遵循以下步骤：
第1步，定义问题：首先，我们需要一个待解决的问题。
第2步，选择合适Rails类型重塑问题：apply_rails函数将问题修改为三者之一，比如主题Rails、事实核查Rails和防止越狱Rails
第3步，大模型生成答案：将递给LLM_response函数，然后返回了一个示例回答
'''
# Step1、准备模拟模型回答
def LLM_response(question):
    # 这是一个示例函数，用于模拟大型语言模型的响应。
    # 在实际应用中，您需要将其替换为实际的大型语言模型API调用。
    return f"示例回答：关于'{question}'，我目前无法提供详细的回答。"

# 本例只构造约束提示；没有实现事实核验或安全防护判定。
# 提示中的“请遵守”不构成结果已满足约束的证据。
# Step2、按约束类型构造提示
def apply_rails(question, rails_type):
    # 根据Rails类型对问题进行处理
    if rails_type == "Topical Rails":
        # 主题 Rails：在提示中要求围绕主题，实际输出仍需校验。
        return LLM_response(f"请仅针对特定主题回答：{question}")
    elif rails_type == "Fact-Checking Rails":
        # 事实核查Rails：尽量减少生成虚假或误导性信息
        return LLM_response(f"请提供一个准确且经过事实核查的回答：{question}")
    elif rails_type == "Jailbreaking Rails":
        # 防止越狱 Rails：仅演示约束措辞，未实现越狱检测。
        return LLM_response(f"请提供一个遵守所有操作限制和指导原则的回答：{question}")
    else:
        # 如果未指定Rails类型，则返回默认回答
        return LLM_response(question)

# 测试代码
# Step3、输入问题并观察约束提示的构造结果
test_question = "什么是人工智能？"
test_rails_type = "Topical Rails"
print(apply_rails(test_question, test_rails_type))
