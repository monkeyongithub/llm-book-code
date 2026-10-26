# 文件用途：演示ToolAgent工具调用
# 对应章节/小节：第8章 8.4.5.1 模型推理——基于LangChain框架并结合GPT-4o以及GPT-4模型实现ToolAgent
# 作者：NYY
# 创建日期：2024-10-30（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""用同一组问题比较四类搜索工具，并用模型给出辅助评分。

评分属于模型判断，不能代替人工核验事实与来源。
参考：https://github.com/darinkist/Enhancing_LLMs_with_Search_APIs
"""

import json
import os
from datetime import date

import requests
from tavily import TavilyClient
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# Step1、为每个搜索服务定义工具，密钥从各自环境变量读取
# T1、You.com 搜索
@tool
def you_com_api(query: str) -> str:
    """使用 You.com 搜索网页并返回来源信息。"""
    response = requests.get("https://api.ydc-index.io/search",
                            params={"query": query, "num_web_results": 3},
                            headers={"X-API-Key": os.environ["YOU_API_KEY"]}, timeout=30)
    response.raise_for_status()
    return json.dumps(response.json(), ensure_ascii=False)


# T2、Exa 搜索
@tool
def exa_search_api(query: str) -> str:
    """使用 Exa 搜索网页并返回来源信息。"""
    response = requests.post("https://api.exa.ai/search",
                             json={"query": query, "contents": {"text": True}, "numResults": 3},
                             headers={"x-api-key": os.environ["EXA_API_KEY"]}, timeout=30)
    response.raise_for_status()
    return json.dumps(response.json(), ensure_ascii=False)


# T3、Tavily 搜索
@tool
def tavily_search_api(query: str) -> str:
    """使用 Tavily 搜索网页并返回来源信息。"""
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    result = client.search(query=query, search_depth="advanced", max_results=3,
                           include_answer=True, include_raw_content=False)
    return json.dumps(result, ensure_ascii=False)


# T4、Perplexity 联网回答（返回合成答案，输出形式与前三种搜索结果不同）
@tool
def perplexity_ai_api(query: str) -> str:
    """通过 Perplexity 获取基于网页资料的回答及引用来源。"""
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        json={"model": os.environ["PERPLEXITY_MODEL"],
              "messages": [{"role": "user", "content": query}]},
        headers={"Authorization": f"Bearer {os.environ['PERPLEXITY_API_KEY']}"}, timeout=60,
    )
    response.raise_for_status()
    return json.dumps(response.json(), ensure_ascii=False)


# Step2、给智能体绑定单个工具，使每轮比较使用明确的搜索服务
def ask_agent(question, search_tool):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "请先使用提供的搜索工具，再依据检索结果回答并列出来源；资料不足时明确说明。"),
        ("user", "{input}"), MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, [search_tool], prompt)
    executor = AgentExecutor(agent=agent, tools=[search_tool], max_iterations=5)
    return executor.invoke({"input": question})["output"]


# Step3、将评估结果解析为 JSON，并校验分数范围
def evaluate_answer(question, answer):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是回答评估员。当前日期为 {today}。仅依据给定问题和回答作辅助评价，"
         "不要把未核验的来源当作已证实事实。输出 JSON，包含 ANSWER（YES或NO）、"
         "DETAIL_RATING、UP_TO_DATE_RATING、QUALITY_RATING（均为1至5的整数），以及 REASONING。"
         "时效性应相对于问题指定的时间判断；缺乏依据时在 REASONING 中说明。"),
        ("user", "问题：{question}\n待评价回答：{answer}"),
    ])
    chain = prompt | ChatOpenAI(model="gpt-4", temperature=0) | JsonOutputParser()
    result = chain.invoke({"today": date.today().isoformat(), "question": question, "answer": answer})
    if not isinstance(result, dict) or result.get("ANSWER") not in ("YES", "NO"):
        raise ValueError("评价结果缺少有效的 ANSWER 字段")
    for key in ("DETAIL_RATING", "UP_TO_DATE_RATING", "QUALITY_RATING"):
        value = result.get(key)
        if type(value) is not int or not 1 <= value <= 5:
            raise ValueError(f"评价字段 {key} 应为 1 至 5 的整数")
    if not isinstance(result.get("REASONING"), str):
        raise ValueError("评价结果缺少文字依据")
    return result


# Step4、汇总答案、评价与失败原因，保留人工复核依据
if __name__ == "__main__":
    questions = ["2024年的主要营销趋势是什么？", "2024年2月发生了哪些重要地缘政治事件？",
                 "最近的人工智能研究有哪些进展？", "美国劳工统计局公布的最新失业率是多少？"]
    results = {}
    for search_tool in [you_com_api, exa_search_api, tavily_search_api, perplexity_ai_api]:
        results[search_tool.name] = {}
        for question in questions:
            try:
                answer = ask_agent(question, search_tool)
                results[search_tool.name][question] = {
                    "answer": answer, "evaluation": evaluate_answer(question, answer)}
            except Exception as error:
                # 一个服务失败不影响其余比较；失败记录不算作低质量得分。
                results[search_tool.name][question] = {"error": str(error)}
    print(json.dumps(results, ensure_ascii=False, indent=2))
