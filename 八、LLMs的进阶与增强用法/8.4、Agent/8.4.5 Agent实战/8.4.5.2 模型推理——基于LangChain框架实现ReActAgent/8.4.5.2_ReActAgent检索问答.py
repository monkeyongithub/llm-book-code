# 文件用途：演示ReAct智能体
# 对应章节/小节：第8章 8.4.5.2 模型推理——基于LangChain框架实现ReActAgent
# 作者：NYY
# 创建日期：2024-10-30（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# 使用LangChain框架构建一个能够与用户进行交互并利用外部工具进行信息检索的ReActAgent智能体


from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI


# Step1、定义搜索工具
# 定义了一个工具列表，其中包含一个TavilySearchResults实例，且该工具被限制为只返回一个搜索结果
tools = [TavilySearchResults(max_results=1)]


# Step2、加载 ReAct 提示模板
prompt = hub.pull("hwchase17/react")


# Step3、选择语言模型
llm = OpenAI()


# Step4、构建 ReAct Agent
# 使用LLM、工具列表和prompt创建一个ReAct智能体。
# ReAct是一种基于反应的智能体，能够根据输入和工具的输出来决定下一步行动。
agent = create_react_agent(llm, tools, prompt)


# Step5、创建执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # verbose=True表示在执行过程中会打印出详细的日志信息


# Step6、测试单轮问答
agent_executor.invoke({"input": "what is LangChain?"})

# 执行过程示例（中文释义；保留 ReAct 的字段名）：
# 进入 AgentExecutor 调用链。
# Thought：需要搜索资料，了解 LangChain。
# Action：tavily_search_results_json；Action Input：LangChain。
# Observation：搜索结果介绍了 LangChain 的用途、模型接口和外部数据集成。
# Thought：进一步了解其功能及集成方式。
# Action：tavily_search_results_json；Action Input：LangChain 的功能和集成方式。
# Observation：搜索结果补充了 embedding、向量存储等集成信息。
# Thought：继续查找项目的发布时间与发展情况。
# Action：tavily_search_results_json；Action Input：LangChain 的发布时间与发展情况。
# Final Answer：LangChain 是用于构建 LLM 应用的开源编排框架，可用于聊天机器人和虚拟助手。
# 调用链结束。
# 返回字典中的 input 保存问题，output 保存最终回答；具体结果取决于检索内容。


# Step7、测试带历史的对话
# 采用不同的prompt——适用于聊天场景
from langchain_core.messages import AIMessage, HumanMessage
prompt = hub.pull("hwchase17/react-chat")


# 重构智能体执行器：智能体执行器被调用时，传入了一个包含用户输入和聊天历史的字典
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 执行智能体：智能体通过聊天历史，确定不需要使用工具，直接给出了最终答案
agent_executor.invoke(
    {
        "input": "what's my name? Only use a tool if needed, otherwise respond with Final Answer",
        # 此提示模板接收文本形式的对话历史，因此 chat_history 应传入字符串
        "chat_history": "Human: Hi! My name is Bob\nAI: Hello Bob! Nice to meet you",
    }
)

# 执行过程示例（中文释义）：
# 进入 AgentExecutor 调用链。
# Thought：对话历史中已有姓名，无需调用工具。
# Final Answer：你的名字是 Bob。
# 调用链结束。
# 返回字典中的 input 保存问题，chat_history 保存输入的历史，output 保存回答。
# chat_history 中的 Human、AI 是角色标记，帮助模型区分用户和助手发言。
