# 文件用途：演示ToolAgent服务部署
# 对应章节/小节：第8章 8.4.5.4 模型部署FastAPI——基于LangChain框架和FastAPI实现部署ToolAgent服务
# 作者：NYY
# 创建日期：2024-10-30（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# !/usr/bin/env python
from typing import List

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes

# Step1、加载网页并创建检索器
# 从指定的网页加载文档。
# 使用文本分割器将文档分割成更小的部分。
# 通过OpenAI嵌入（Embeddings）为分割后的文本创建向量。
# 使用FAISS创建一个向量存储，并从中创建一个检索器。

# 加载检索器
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# Step2、封装检索与在线搜索工具
# 创建工具
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)
search = TavilySearchResults()
tools = [retriever_tool, search]

# Step3、建立 Agent 执行器
# 从一个中心仓库（hub）获取代理的提示（prompt）模板。
# 初始化一个ChatOpenAI实例，使用GPT-3.5模型。
# 使用OpenAI函数代理创建函数，结合LLM、工具和提示。
# 创建一个代理执行器（AgentExecutor），用于执行代理。

# 创建 Agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Step4、创建 FastAPI 应用
# 创建一个FastAPI应用实例，并设置其标题、版本和描述。
# 定义应用
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)


# Step5、定义请求与响应并注册路由
# 定义输入和输出模型，用于处理用户输入和提供代理执行结果。
# 添加一个路由到FastAPI应用，将代理执行器连接到特定的URL路径。

# 添加路由
# 为 AgentExecutor 补充明确的输入、输出模式，
# 供 LangServe 生成接口定义并处理请求。

class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        default_factory=list,
        description="当前会话的历史消息",
    )


class Output(BaseModel):
    output: str


add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)

if __name__ == "__main__":
    # Step6、启动服务
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
