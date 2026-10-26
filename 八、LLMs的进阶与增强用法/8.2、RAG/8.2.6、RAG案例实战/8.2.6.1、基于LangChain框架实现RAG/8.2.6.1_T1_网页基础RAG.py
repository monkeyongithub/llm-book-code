# 文件用途：演示网页数据RAG检索链
# 对应章节/小节：第8章 8.2.6.1 基于LangChain框架实现RAG
# 作者：NYY
# 创建日期：2024-10-29（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# 首先，我们需要加载我们想要索引的数据。为此，我们将使用 WebBaseLoader。这需要安装 BeautifulSoup：
# pip install beautifulsoup4


import os

from langchain_core.prompts import ChatPromptTemplate
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

# Step2、比较直接回答与检索增强回答
# 接收一个问题→查找相关的文档→将这些文档与原始问题一起传递给语言模型→模型回答原始问题
# 现在我们已经将这个数据索引到向量存储中，我们将创建一个检索链。这个链将接收一个传入的问题，查找相关文档，然后将这些文档连同原始问题一起传递给 LLM 并请它回答原始问题。
# 首先，让我们设置一个链，它接收一个问题以及检索到的文档并生成答案。
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)

# T1、直接回答：给定文本实现
# 如果我们愿意，我们可以通过直接传递文档来自己运行这个：
from langchain_core.documents import Document

document_chain.invoke({
    "input": "how can langsmith help with testing?",
    "context": [Document(page_content="langsmith can let you visualize test results")]
})

# T2、检索增强回答：基于网页文档检索相关内容后实现
# 然而，我们希望文档首先来自我们刚刚设置的检索器。这样，我们可以使用检索器动态选择最相关的文档并传递给给定的问题。
from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])
# 回答示例（中文释义）：LangSmith 提供了多种辅助测试的功能……

# 检索质量和资料内容影响答案，需结合返回的来源核对。
