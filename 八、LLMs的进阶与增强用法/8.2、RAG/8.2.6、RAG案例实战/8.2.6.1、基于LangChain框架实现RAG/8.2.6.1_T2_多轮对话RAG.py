# 文件用途：演示多轮对话RAG检索链
# 对应章节/小节：第8章 8.2.6.1 基于LangChain框架实现RAG
# 作者：NYY
# 创建日期：2024-10-29（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

import os

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 背景资料内容
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=os.environ["OPENAI_API_KEY"])

# Step1、加载网页并建立向量检索器
# 采用WebBaseLoader工具从指定网页上加载数据
# 数据索引到一个向量存储器中以便后续检索使用
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
print(docs)

# 接下来，我们需要将其索引到向量存储中。这需要几个组件，即嵌入模型和向量存储。
# 对于嵌入模型，我们再次提供通过 API 访问或运行本地模型的例子。
# 确保您已安装 langchain_openai 包并设置了适当的环境变量（这些与 LLM 需要的相同）
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 嵌入模型(OpenAIEmbeddings)将文档向量化并存储存到FAISS
# 现在，我们可以使用这个嵌入模型将文档摄取到向量存储中。为了简单起见，我们将使用一个简单的本地向量存储，FAISS。
# 首先，我们需要安装所需的包：
# pip install faiss-cpu
# 然后我们可以构建我们的索引：
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# 本脚本演示基于聊天历史的检索链。
retriever = vector.as_retriever()

###-------------------------------------新增内容分界线--------------------------------------------###

# Step2、构建考虑聊天历史的检索器
# 这个链将接收最近的输入（input）和对话历史（chat_history）,告诉LLM基于上述对话生成一个相关的搜索查询
# 设置对话模板prompt，包括chat_history+input，这是一个可以传递给 LLM 以生成搜索查询的提示
# 创建了一个基于对话历史的信息检索器链。
# 定义提示模板，让 LLM 根据对话历史生成检索查询
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user",
     "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    # 给定上述对话，生成一个搜索查询以查找与对话相关的信息
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# Step3、检查历史感知检索结果
# 提供一个对话历史的实例和一个后续问题，然后调用检索链，观察返回的文档是否与测试相关
# 构建chat_history：包括了用户的消息和AI的回复
from langchain_core.messages import HumanMessage, AIMessage

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"),
                AIMessage(content="Yes!")]

# 测试检索器链：使用创建的检索器链，将对话历史和用户输入传递给检索器，以获取相关信息。
res = retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
print(res)
# 您应该看到这返回了关于 LangSmith 测试的文档。这是因为 LLM 生成了一个新的查询，将对话历史与后续问题结合起来。


# Step4、构建文档问答链
# 定义基于对话历史的ChatPromptTemplate模板，包括了对话历史和用户输入的占位符，用于指导LLM基于提供的上下文和对话历史来回答用户的问题
# 创建了一个处理文档的链，用于处理从检索器获取的文档信息，以便在考虑到这些检索到的文档的情况下继续对话。
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)

# Step5、组合完整检索链
# 将之前创建的对话历史感知的检索器链和文档处理链连接起来，以实现更复杂的信息检索功能。
from langchain.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

# Step6、利用聊天历史测试问答
chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
res = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
print(res)
# 我们可以看到这给出了一个连贯的回答 - 我们成功地将我们的检索链变成了一个聊天机器人！
