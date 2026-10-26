# 文件用途：演示链式提示
# 对应章节/小节：第7章 7.3.5 提示设计方法论分类
# 作者：NYY
# 创建日期：2024-03-29（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

#######
'''
使用Python编程语言实现一个名为"Chains"的链式处理流程，使用大型语言模型（LLMs）来处理复杂的任务。
为了实现这个策略，我们将遵循以下步骤：
每个链式组件执行特定的功能，并将输出作为下一个组件的输入。
我们将使用一个假设的LLM_response(question)函数来代表大模型的回答功能。这个函数将接收一个问题作为输入，并返回模型的回答。
第1步，使用chain_component_1函数处理输入数据并生成一个问题，
第2步，使用chain_component_2函数利用大型语言模型来回答这个问题。
'''


def LLM_response(question):
    # 这个函数模拟了一个大型语言模型的回答功能。
    # 在实际应用中，这将是调用大型语言模型的接口。
    # 为了示例，我们简单地返回一个格式化的回答。
    return f"模型对'{question}'的回答是：这里会插入模型的回答。"

# Step1、把输入转换为问题
def chain_component_1(input_data):
    # 第一个链式组件，它将处理输入数据并生成一个问题。
    # 在实际应用中，这可能包括数据预处理、问题生成等步骤。
    question = f"请解释'{input_data}'的含义。"
    return question

# Step2、将问题交给模拟模型回答
def chain_component_2(input_data):
    # 第二个链式组件，它将使用大型语言模型来回答第一个组件生成的问题。
    response = LLM_response(input_data)
    return response

# Step3、顺序连接两个提示组件
def chain_of_components(input_data):
    # 链式处理流程，将依次调用每个组件。
    question = chain_component_1(input_data)
    answer = chain_component_2(question)
    return answer

# 示例输入数据
example_input = "Chains"

# 运行链式处理流程
chain_result = chain_of_components(example_input)
print(chain_result)
