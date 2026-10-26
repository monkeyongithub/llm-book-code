# 文件用途：演示LlamaIndex RAG
# 对应章节/小节：第8章 8.2.6.3 基于LLaMAIndex框架实现RAG
# 作者：NYY
# 创建日期：2024-10-29（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

'''
在线地址：https://colab.research.google.com/github/Crossme0809/langchain-tutorials/blob/main/Using_LlamaIndex_Query_Documents.ipynb#scrollTo=VImNM7jLUQHp
'''

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

import os

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("请先设置 OPENAI_API_KEY")

# Step1、读取文档并建立向量索引
# 利用SimpleDirectoryReader数据加载器，加载整个目录，里面的每一个文件，都会被当成是一篇文档。
path = 'data_txt_doc/temp_test'
documents = SimpleDirectoryReader(path).load_data()
# 利用GPTSimpleVectorIndex对所有文档构建索引(基于ChatGPT接口)：它会把文档分段转换成一个个向量，然后存储成一个索引。
index = VectorStoreIndex.from_documents(documents)
index.set_index_id("index_id_demo")
index.storage_context.persist('./storage')
# 保存对应的索引为一个json文件，后面就可以用这个索引来进行相应的问答。
# 注：日志中会打印出来我们通过 Embedding 消耗了多少个 Token。
# 日志示例（中文释义）：构建索引时，LLM 总计使用 0 个 token。
# 日志示例（中文释义）：构建索引时，embedding 总计使用 6763 个 token。

from llama_index.core import StorageContext, load_index_from_storage

# Step2、持久化并重新载入索引
storage_context = StorageContext.from_defaults(persist_dir='./storage')
index = load_index_from_storage(storage_context, index_id="index_id_demo")

# Step3、执行索引查询
query_engine = index.as_query_engine(response_mode="tree_summarize")
response = query_engine.query("根据上下文内容，告诉我鲁迅先生在日本学医的老师是谁?")
print(response)
response = query_engine.query("根据上下文内容，鲁迅先生是去哪里学的医学？")
print(response)
# 日志示例（中文释义）：查询时，LLM 总计使用 2984 个 token。
# 日志示例（中文释义）：查询时，embedding 总计使用 34 个 token。
# 鲁迅先生在日本学习医学的老师是藤野严九郎先生。
# 检索到的节点文本：藤野先生
# 东京也无非是这样。上野的樱花烂熳的时节，望去确也像绯红的轻云，但花下也缺不了成群结队的“清国留学生”的速成班，头顶上盘着大辫子，顶得学生制帽的顶上高高耸起，形成一座富士山。也有解散辫子，盘得平的，除下帽来，油光可鉴，宛如小姑娘的发髻一般，还要将脖子扭几扭。实在标致极了。
# 中国留学生会馆的门房里有几本书买，有时还值得去一转；倘在上午，里面的几间洋房里倒也还可以坐坐的。但到傍晚，有...
# 日志示例（中文释义）：查询时，LLM 总计使用 2969 个 token。
# 日志示例（中文释义）：查询时，embedding 总计使用 26 个 token。
# 鲁迅先生去仙台的医学专门学校学习医学。

# Step4、比较不同提示模板下的问答
# 把搜索找到的相关内容以及问题，组合到一起变成一段提示语，让 AI 能够按照我们的要求来回答问题
from llama_index.core import PromptTemplate

query_str = "鲁迅先生去哪里学的医学？"

# T1、定义提示模板01
# 模板开头，来告诉AI，为AI提供了一些上下文信息
# 支持两个变量context_str、query_str。context_str：在实际调用的时候，会被通过 Embedding 相似度找出来的内容填入。而query_str则是会被我们实际提的问题替换掉。
# 实际提问的时候，告诉AI，只考虑上下文信息，而不要根据自己已经有的先验知识（prior knowledge）来回答问题
Default_Text_QA_Prompt_Tmpl = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
)
QA_PROMPT = PromptTemplate(Default_Text_QA_Prompt_Tmpl)
query_engine = index.as_query_engine(text_qa_template=QA_PROMPT)
response = query_engine.query(query_str)
print(response)
# 鲁迅先生去仙台的医学专门学校学习医学

# T2、定义提示模板02：并问一些不相干的问题
QA_PROMPT_TMPL = (
    "下面的“我”指的是鲁迅先生 \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "根据这些信息，请回答问题: {query_str}\n"
    "如果您不知道的话，请回答不知道\n"
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)
query_engine = index.as_query_engine(text_qa_template=QA_PROMPT)
response = query_engine.query("请问西游记的作者是谁？")
print(response)
# 不知道
