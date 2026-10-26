# 文件用途：演示知识图谱RAG智能体
# 对应章节/小节：第8章 8.4.5.3 模型推理——基于LangChain框架实现KG-RAG-Agent
# 作者：NYY
# 创建日期：2024-10-30（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# LLMs之KG-RAG-Agent：基于LangChain框架利用知识图谱(Neo4j构建知识图数据库+Cyphe查询语言)结合RAG技术打造多种工具的Agent进而实现与图知识库对话查询的机器人应用实战案例——导入知识图谱数数据(Neo4j)→基于向量索引和OpenAIEmbeddings构建RetrievalQA并测试相似性搜索→创建一个GraphCypherQAChain的Cyphe图查询式问答系统并并使用知识图谱测试→创建多模态Agent并测试(定义一个工具列表【Tasks/Graph】+创建一个多模态Agent+测试多模态Agent回答问题)
'''
!pip install neo4j openai langchain tiktoken langchain_openai langchainhub
博客文章地址：https://neo4j.com/developer-blog/knowledge-graph-rag-application
源代码地址：https://github.com/tomasonjo/blogs/blob/master/llm/devops_rag.ipynb
'''

# 案例目标：创建一个能够处理结构化和非结构化数据的聊天机器人，特别针对DevOps场景。

# Step1、准备示例知识

# Step2、配置 Neo4j 环境
# 设置Neo4j数据库实例，可以是本地实例或使用Neo4j Aura等云服务。使用凭据（URL、用户名、密码）创建与Neo4j数据库的连接。
# 需要设置一个 Neo4j 5.11 实例或更高版本，以便按照示例进行操作。
# T1、最简单的方法是在Neo4j Aura上启动 Neo4j 数据库的免费云实例。
# T2、可以通过下载Neo4j Desktop应用程序并创建本地数据库实例来设置 Neo4j 数据库的本地实例。

import os
from langchain_community.graphs import Neo4jGraph
neo4j_url = os.environ["NEO4J_URI"]
username = os.environ.get("NEO4J_USERNAME", "neo4j")
password = os.environ["NEO4J_PASSWORD"]

graph = Neo4jGraph(
    url=neo4j_url,
    username=username,
    password=password
)



# Step3、导入图谱数据
# 数据集：知识图非常擅长连接来自多个数据源的信息。开发 DevOps RAG 应用程序时，您可以从云服务、任务管理工具等获取信息。
# 在Neo4j 浏览器中看到类似的图形可视化
# 在知识图谱中，节点和关系用于描述数据。通常，节点用来表示实体或概念，如人物、组织和地点。
# 在微服务图示例中，节点描述了人物、团队、微服务和任务。另一方面，关系则用于定义这些实体之间的连接，比如微服务之间的依赖性或任务的所有者。
# 节点和关系都可以存储属性值作为键值对。
# 微服务节点有两个节点属性来描述它们的名字和技术。相比之下，任务节点更为复杂。它们有名称、状态、描述以及嵌入属性。通过将文本嵌入值存储为节点属性，你可以执行向量相似度搜索，就像你在向量数据库中存储任务一样。
# 因此，知识图谱允许你存储和检索结构化及非结构化信息，为你的RAG应用程序提供支持。

# DevOps 图的的解读
# 蓝色节点描述微服务。这些微服务可能相互依赖。这意味着一个微服务的运行或提供结果的能力可能依赖于另一个微服务的操作。
# 棕色节点代表直接链接到这些微服务的任务。我们的图表示例一起展示了微服务的设置方式、它们的链接任务以及与每个服务相关的团队。
import requests

url = "https://gist.githubusercontent.com/tomasonjo/08dc8ba0e19d592c4c3cde40dd6abcc3/raw/da8882249af3e819a80debf3160ebbb3513ee962/microservices.json"
response = requests.get(url, timeout=30)
response.raise_for_status()
import_query = response.json()["query"]
graph.query(
    import_query
)


# Step4、建立向量检索问答
# 向量索引：使用 OpenAI 的嵌入模型为任务节点在Neo4j中创建向量索引，以支持基于内容的相似性搜索。计算任务描述的嵌入向量。
# 首先实现向量索引搜索，通过名称和描述查找相关任务
# 关键思想是根据每个任务的描述和名称计算文本嵌入值。然后，在查询时，使用余弦距离等相似性度量找到与用户输入最相似的任务。
# 这些任务已经在我们的知识图中了。但是，我们必须计算嵌入值并创建向量索引。在这里，我们将使用 from_existing_graph 方法
import os
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("请先设置 OPENAI_API_KEY")
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    url=neo4j_url,
    username=username,
    password=password,
    index_name='tasks',
    node_label="Task",
    text_node_properties=['name', 'description', 'status'],
    embedding_node_property='embedding',
)


# 基于文本嵌入通过相似性搜索查询知识图谱
response = vector_index.similarity_search(
    "How will RecommendationService be updated?"
)
print(response[0].page_content if response else "没有检索结果")
# 检索结果示例（中文释义）：name 为 BugFix。
# description：为 RecommendationService 增加新功能，以提供……
# status：进行中，对应数据中的 In Progress。


# 基于ChatOpenAI创建一个RetrievalQA检索式问答系统并使用向量存储来回答问题
# 将向量索引包装到 RetrievalQA 模块中并使用向量存储来回答问题
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

vector_qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=vector_index.as_retriever()
)
# 根据用户查询生成响应
vector_qa.run(
    "How will recommendation service be updated?"
)
# 回答示例（中文释义）：RecommendationService 正在增加新功能，
# 利用用户行为和偏好数据改进推荐算法，
# 为用户提供更个性化、更准确的商品推荐；当前任务状态为进行中。

vector_qa.run(
    "How many open tickets are there?"
)
# 回答示例（中文释义）：有 4 个未关闭的工单；下方用 Cypher 核验计数。


# 查询机制：利用向量索引进行快速的内容匹配。使用Cypher语言执行结构化查询，以支持更复杂的逻辑和数据聚合。
# 可以使用 Neo4j的查询语言Cypher 语句验证该搜索结果是否正确
graph.query(
    "MATCH (t:Task {status:'Open'}) RETURN count(*)"
)
# [{'count(*)': 5}]
# 向量索引的一个普遍限制是它们不提供聚合信息的能力，就像使用 Cypher 等结构化查询语言一样。
# 上述结果可知，向量相似性搜索非常适合筛选非结构化文本中的相关信息，但缺乏分析和聚合结构化信息的能力


# Step5、建立 Cypher 图查询问答
# 使用Cypher进行数据聚合，并回答需要结构化分析的问题。
# Cypher 是一种结构化查询语言，旨在与图数据库交互。它提供了一种匹配模式和关系的可视化方式
# LangChain 的巧妙之处在于它提供了GraphCypherQAChain，它可以为您生成 Cypher 查询，因此您无需学习 Cypher 语法即可从 Neo4j 等图形数据库中检索信息。
# (:Person {name:"Tomaz"})-[:LIVES_IN]->(:Country {name:"Slovenia"})

from langchain.chains import GraphCypherQAChain

graph.refresh_schema()
# 生成有效的 Cypher 语句是一项复杂的任务。因此，建议使用 gpt-4 等最先进的 LLM 来生成 Cypher 语句，而使用数据库上下文生成答案可以留给 gpt-3.5-turbo。
# 将向量索引和Cypher查询的结果传递给语言模型，以生成最终的回复
cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(temperature=0, model_name='gpt-4'),
    qa_llm=ChatOpenAI(temperature=0), graph=graph, verbose=True,
    # 本例会导入数据并执行生成的 Cypher，请连接专用教学数据库。
    allow_dangerous_requests=True,
)

cypher_chain.run(
    "How many open tickets there are?"
)

cypher_chain.run(
    "Which team has the most open tasks?"
)

cypher_chain.run(
    "Which services depend on Database directly?"
)

cypher_chain.run(
    "Which services depend on Database indirectly?"
)

# Step6、组合两类工具并测试 Agent
# 定义一个工具列表，包括 Tasks(非结构化数据)和 Graph(知识图谱的结构化数据)，这些工具可以用于回答不同类型的问题
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

tools = [
    Tool(
        name="Tasks",
        func=vector_qa.run,
        description="""Useful when you need to answer questions about descriptions of tasks.
        Not useful for counting the number of tasks.
        Use full question as input.
        """,
    ),
    Tool(
        name="Graph",
        func=cypher_chain.run,
        description="""Useful when you need to answer questions about microservices,
        their dependencies or assigned people. Also useful for any sort of 
        aggregation like counting the number of tasks, etc.
        Use full question as input.
        """,
    ),
]

# 创建一个多模态Agent：基于tools组合和OpenAI的语言模型，可以根据问题的类型选择合适的工具来回答。
# 代理构建：构建一个智能代理，可以根据查询类型自动选择适当的工具（向量索引或Cypher查询）
# "mrkl" 是一个缩写，它代表 "MarkLogic"，即Neo4j，这是一种常见的 NoSQL的图数据库，用于存储和管理半结构化和非结构化数据。
mrkl = initialize_agent(
    tools,
    ChatOpenAI(temperature=0, model_name='gpt-4'),
    agent=AgentType.OPENAI_FUNCTIONS, verbose=True
)


# 测试多模态Agent回答问题
# 测试整个系统的性能，并调整代理提示以提高用户体验
# 探索知识图谱工具
response = mrkl.run("Which team is assigned to maintain PaymentService?")
print(response)
# 调用任务工具
response = mrkl.run("Which tasks have optimization in their description?")
print(response)
