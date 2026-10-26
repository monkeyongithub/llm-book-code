# 文件用途：演示ChatGLM-3工具调用
# 对应章节/小节：第5章 5.2.2.3 采用基于LangChain框架实现ChatGLM3的工具调用功能
# 作者：NYY
# 创建日期：2024-10-11（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# 本节示例：采用基于LangChain框架实现ChatGLM3的工具调用功能

'''
源码地址：https://github.com/THUDM/ChatGLM3/blob/main/langchain_demo/main.py
'''

import os

from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent, load_tools
from langchain_core.messages import AIMessage, HumanMessage

# 以下模块来自书稿所用 ChatGLM3/langchain_demo 项目，需将该目录加入 PYTHONPATH。
from ChatGLM3 import ChatGLM3
from tools.Calculator import Calculator
from tools.Weather import Weather
from tools.DistanceConversion import DistanceConverter

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')

# Step1、创建 ChatGLM-3 与结构化工具提示
if __name__ == "__main__":
    llm = ChatGLM3()
    llm.load_model(MODEL_PATH)
    prompt = hub.pull("hwchase17/structured-chat-agent")

    # 单参数工具调用：不传入对话历史

    # Step2、比较计算、天气和单位换算工具
    tools = [Calculator()]
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    ans = agent_executor.invoke({"input": "34 * 34"})
    print(ans)

    # 单参数工具调用：传入对话历史

    tools = [Weather()]
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    ans = agent_executor.invoke(
        {
            "input": "厦门比北京热吗?",
            "chat_history": [
                HumanMessage(content="北京温度多少度"),
                AIMessage(content="北京现在12度"),
            ],
        }
    )
    print(ans)

    # 多参数工具调用：不传入对话历史

    tools = [DistanceConverter()]
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    ans = agent_executor.invoke({"input": "how many meters in 30 km?"})

    print(ans)

    # 使用 LangChain 内置工具

    # Step3、调用 LangChain 内置检索工具
    tools = load_tools(["arxiv"], llm=llm)
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    ans = agent_executor.invoke({"input": "Describe the paper about GLM 130B"})

    print(ans)
