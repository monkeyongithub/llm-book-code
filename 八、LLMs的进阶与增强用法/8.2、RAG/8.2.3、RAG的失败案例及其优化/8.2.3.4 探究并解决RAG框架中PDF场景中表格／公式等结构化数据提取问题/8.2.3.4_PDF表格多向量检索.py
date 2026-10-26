# 文件用途：演示PDF表格解析和检索
# 对应章节/小节：第8章 8.2.3.4 探究并解决RAG框架中PDF场景中表格/公式等结构化数据提取问题
# 作者：NYY
# 创建日期：2024-10-30（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# LLMs之RAG：如何正确解析企业内部如PDF类型中结构化数据——利用LangChain框架的MultiVectorRetriever函数解决PDF中的Table类型数据进而实现文档问答代码实战(对table信息单独做专门的保存+解码+embedding处理)

from lxml import html
from pydantic import BaseModel
from typing import Any, Optional
from unstructured.partition.pdf import partition_pdf


# Step1、解析 PDF 元素并统计类别
# # 该步骤由于涉及到OCR，导致时间很长，在 colab 上花了14分钟，77页面平均11s/p
# 提取 PDF 中的文本、表格等元素
pdf_path = "2307.09288.pdf"
raw_pdf_elements = partition_pdf(filename=pdf_path,
                                 # 不单独提取 PDF 中的图像块
                                 extract_images_in_pdf=False,
                                 # 通过版面分析识别表格，并推断表格结构
                                 # 识别出的标题用于后续按章节组织文本块
                                 infer_table_structure=True,
                                 # 按标题聚合文本元素
                                 chunking_strategy="by_title",
                                 # 设置文本块的字符数上限与合并阈值
                                 # 超过 3800 个字符后尝试开始新块
                                 # 尝试合并不足 2000 个字符的文本块
                                 max_characters=4000,
                                 new_after_n_chars=3800,
                                 combine_text_under_n_chars=2000)
# 用字典统计各类元素的数量
category_counts = {}
for element in raw_pdf_elements:
    category = str(type(element))
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1
# 提取不重复的元素类别
unique_categories = set(category_counts.keys())
print(category_counts)
# { "<class 'unstructured.documents.elements.CompositeElement '>": 120,
#  "<class 'unstructured.documents.elements.Table '>": 47,
#  "<class 'unstructured.documents.elements.TableChunk '>": 2,

# Step2、分离正文与表格
class Element(BaseModel):
    type: str
    text: Any
# 按元素类型分别整理表格和正文
categorized_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        categorized_elements.append(Element(type="table", text=str(element)))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        categorized_elements.append(Element(type="text", text=str(element)))
# 筛选表格元素
table_elements = [e for e in categorized_elements if e.type == "table"]
print(len(table_elements))
# 筛选正文元素
text_elements = [e for e in categorized_elements if e.type == "text"]
print(len(text_elements))
# 输出示例：
# 49
# 120

# Step3、构建摘要链
# 引入了 langchain-hub 里面一个社区贡献的 prompt
# 多向量检索摘要提示的中文释义：
# 你负责为表格和正文生成摘要。
# 请简明概括以下表格或文本块：{element}
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
obj = hub.pull("rlm/multi-vector-retriever-summarization")
# 将提示模板、模型和输出解析器串成摘要链
prompt = ChatPromptTemplate.from_template(obj.template)
model = ChatOpenAI(temperature=0 ,model="gpt-4")
summarize_chain = {"element": lambda x :x} | prompt | model | StrOutputParser()

# Step4、分别提取正文和表格摘要
# 为正文批量生成摘要
texts = [i.text for i in text_elements]
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
# 为表格批量生成摘要
tables = [i.text for i in table_elements]
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
# 测试table信息提取总结的效果
print(len(table_summaries))
if table_summaries:
    print(table_summaries[0])
# 表格摘要示例（中文释义，具体数值须与表格原文核对）：
# 表格列出了不同规模模型的训练时间、功耗和碳排放。
# 摘要中的碳排放范围为 31.22 至 291.42 tCO2eq。
# LLaMA 2 的碳排放总量为 539.00 tCO2eq。

# Step5、建立多向量检索器
import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
# 使用向量存储为摘要建立索引
vectorstore = Chroma(
    collection_name="summaries",
    embedding_function=OpenAIEmbeddings()
)
# 使用内存存储保存摘要对应的完整正文和表格
store = InMemoryStore()
id_key = "doc_id"
# 创建检索器，随后写入摘要及其对应的原始文档
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)
# 写入正文摘要及原始正文
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [Document(page_content=s ,metadata={id_key: doc_ids[i]}) for i, s in enumerate(text_summaries)]
if summary_texts:
    retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, [Document(page_content=text) for text in texts])))
# 写入表格摘要及原始表格
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [Document(page_content=s ,metadata={id_key: table_ids[i]}) for i, s in enumerate(table_summaries)]
if summary_tables:
    retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, [Document(page_content=table) for table in tables])))

# Step6、利用检索内容回答表格问题
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough
# 定义问答提示模板
template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
# 配置生成回答的 LLM
model = ChatOpenAI(temperature=0)
# 构建 RAG 问答流程
chain = (
        {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)
# 查询表里面的信息
print(chain.invoke("llama2 70B模型用了多少gpu小时来完成训练?"))
# 回答示例（中文释义）：根据提供的上下文，LLaMA 2 70B 的训练累计使用了 330 万 GPU 小时。
