# 第8章 LLM的进阶与增强用法

本章介绍 RAG、外部工具、Agent 与长上下文。代码展示从检索、工具参数生成到执行结果回传的主要流程，并保留相关框架的操作说明。

<!-- CODE_OVERVIEW:START -->
## 子章节与配套代码

| 书稿小节 | 代码文件 | 用途 |
| --- | --- | --- |
| 8.2.3.3 RAG微观+策略角度优化 | [`8.2、RAG/8.2.3、RAG的失败案例及其优化/8.2.3.3 RAG微观+策略角度优化/8.2.3.3_文本分块演示.py`](<8.2、RAG/8.2.3、RAG的失败案例及其优化/8.2.3.3 RAG微观+策略角度优化/8.2.3.3_文本分块演示.py>) | 演示文本分块 |
| 8.2.3.4 探究并解决RAG框架中PDF场景中表格/公式等结构化数据提取问题 | [`8.2、RAG/8.2.3、RAG的失败案例及其优化/8.2.3.4 探究并解决RAG框架中PDF场景中表格／公式等结构化数据提取问题/8.2.3.4_PDF表格多向量检索.py`](<8.2、RAG/8.2.3、RAG的失败案例及其优化/8.2.3.4 探究并解决RAG框架中PDF场景中表格／公式等结构化数据提取问题/8.2.3.4_PDF表格多向量检索.py>) | 提取 PDF 文本和表格，以摘要检索并返回原始 Document |
| 8.2.6.1 基于LangChain框架实现RAG | [`8.2、RAG/8.2.6、RAG案例实战/8.2.6.1、基于LangChain框架实现RAG/8.2.6.1_T1_网页基础RAG.py`](<8.2、RAG/8.2.6、RAG案例实战/8.2.6.1、基于LangChain框架实现RAG/8.2.6.1_T1_网页基础RAG.py>) | 演示网页数据RAG检索链 |
| 8.2.6.1 基于LangChain框架实现RAG | [`8.2、RAG/8.2.6、RAG案例实战/8.2.6.1、基于LangChain框架实现RAG/8.2.6.1_T2_多轮对话RAG.py`](<8.2、RAG/8.2.6、RAG案例实战/8.2.6.1、基于LangChain框架实现RAG/8.2.6.1_T2_多轮对话RAG.py>) | 演示多轮对话RAG检索链 |
| 8.2.6.1 基于LangChain框架实现RAG | [`8.2、RAG/8.2.6、RAG案例实战/8.2.6.1、基于LangChain框架实现RAG/8.2.6.1_T3_本地PDF混合检索问答.py`](<8.2、RAG/8.2.6、RAG案例实战/8.2.6.1、基于LangChain框架实现RAG/8.2.6.1_T3_本地PDF混合检索问答.py>) | 对比向量、中文 BM25 与排名融合，选择本地聊天模型回答 |
| 8.2.6.2 基于LangChain-ChatGLM框架实现RAG | [`8.2、RAG/8.2.6、RAG案例实战/8.2.6.2、基于LangChain-ChatGLM框架实现RAG/8.2.6.2_T1_Chatchat本地知识库.py`](<8.2、RAG/8.2.6、RAG案例实战/8.2.6.2、基于LangChain-ChatGLM框架实现RAG/8.2.6.2_T1_Chatchat本地知识库.py>) | 说明本地知识库部署 |
| 8.2.6.2 基于LangChain-ChatGLM框架实现RAG | [`8.2、RAG/8.2.6、RAG案例实战/8.2.6.2、基于LangChain-ChatGLM框架实现RAG/8.2.6.2_T2_Chatchat-WebUI部署.py`](<8.2、RAG/8.2.6、RAG案例实战/8.2.6.2、基于LangChain-ChatGLM框架实现RAG/8.2.6.2_T2_Chatchat-WebUI部署.py>) | 说明知识库Web部署 |
| 8.2.6.3 基于LLaMAIndex框架实现RAG | [`8.2、RAG/8.2.6、RAG案例实战/8.2.6.3、基于LLaMAIndex框架实现RAG/8.2.6.3_LlamaIndex检索问答.py`](<8.2、RAG/8.2.6、RAG案例实战/8.2.6.3、基于LLaMAIndex框架实现RAG/8.2.6.3_LlamaIndex检索问答.py>) | 演示LlamaIndex RAG |
| 8.2.6.4 基于LocalGPT框架实现RAG | [`8.2、RAG/8.2.6、RAG案例实战/8.2.6.4、基于LocalGPT框架实现RAG/8.2.6.4_LocalGPT本地知识库.py`](<8.2、RAG/8.2.6、RAG案例实战/8.2.6.4、基于LocalGPT框架实现RAG/8.2.6.4_LocalGPT本地知识库.py>) | 说明LocalGPT RAG部署 |
| 8.2.6.5 基于OLLaMA+AnythingLLM框架实现RAG | [`8.2、RAG/8.2.6、RAG案例实战/8.2.6.5、基于AnythingLLM框架实现RAG/8.2.6.5_AnythingLLM知识库.py`](<8.2、RAG/8.2.6、RAG案例实战/8.2.6.5、基于AnythingLLM框架实现RAG/8.2.6.5_AnythingLLM知识库.py>) | 说明Ollama与AnythingLLM RAG部署 |
| 8.2.6.6 基于OLLaMA后端+Dify前端框架实现RAG | [`8.2、RAG/8.2.6、RAG案例实战/8.2.6.6、基于Dify框架实现RAG/8.2.6.6_Dify知识库应用.py`](<8.2、RAG/8.2.6、RAG案例实战/8.2.6.6、基于Dify框架实现RAG/8.2.6.6_Dify知识库应用.py>) | 说明Ollama与Dify RAG部署 |
| 8.3.2.1 基于OpenAI官方案例实现工具调用能力 | [`8.3、ETA/8.3.2、ETA实战/8.3.2.1 基于OpenAI官方案例实现工具调用能力/8.3.2.1_OpenAI工具调用.py`](<8.3、ETA/8.3.2、ETA实战/8.3.2.1 基于OpenAI官方案例实现工具调用能力/8.3.2.1_OpenAI工具调用.py>) | 演示OpenAI工具调用 |
| 8.3.2.2 基于ChatGLM4官方案例实现工具调用能力 | [`8.3、ETA/8.3.2、ETA实战/8.3.2.2 基于ChatGLM4官方案例实现工具调用能力/8.3.2.2_ChatGLM4工具调用.py`](<8.3、ETA/8.3.2、ETA实战/8.3.2.2 基于ChatGLM4官方案例实现工具调用能力/8.3.2.2_ChatGLM4工具调用.py>) | 模拟航班查询；校验日期并处理多个关联工具调用 |
| 8.3.2.3 基于Qwen官方案例实现工具调用能力 | [`8.3、ETA/8.3.2、ETA实战/8.3.2.3 基于Qwen官方案例实现工具调用能力/8.3.2.3_Qwen2工具调用.py`](<8.3、ETA/8.3.2、ETA实战/8.3.2.3 基于Qwen官方案例实现工具调用能力/8.3.2.3_Qwen2工具调用.py>) | 以本地时间和模拟天气演示多轮工具调用及结果回传 |
| 8.3.2.4 基于Langchain框架利用GPT-4o实现多个工具调用 | [`8.3、ETA/8.3.2、ETA实战/8.3.2.4 基于Langchain框架利用GPT-4o实现多个工具调用/8.3.2.4_LangChain多工具调用.py`](<8.3、ETA/8.3.2、ETA实战/8.3.2.4 基于Langchain框架利用GPT-4o实现多个工具调用/8.3.2.4_LangChain多工具调用.py>) | 比较函数、Pydantic、TypedDict 模式，校验并执行加法与乘法 |
| 8.3.2.5 基于LangGraph框架实现GraphETA | [`8.3、ETA/8.3.2、ETA实战/8.3.2.5 基于LangGraph框架实现GraphETA/8.3.2.5_LangGraph工具调用图.py`](<8.3、ETA/8.3.2、ETA实战/8.3.2.5 基于LangGraph框架实现GraphETA/8.3.2.5_LangGraph工具调用图.py>) | 模型与工具之间循环路由，由模型结合工具结果给出最终回答 |
| 8.4.5.1 模型推理——基于LangChain框架并结合GPT-4o以及GPT-4模型实现ToolAgent | [`8.4、Agent/8.4.5 Agent实战/8.4.5.1 模型推理——基于LangChain框架并结合GPT-4o以及GPT-4模型实现ToolAgent/8.4.5.1_ToolAgent搜索工具比较.py`](<8.4、Agent/8.4.5 Agent实战/8.4.5.1 模型推理——基于LangChain框架并结合GPT-4o以及GPT-4模型实现ToolAgent/8.4.5.1_ToolAgent搜索工具比较.py>) | 比较四类搜索服务，保存回答、模型辅助评分与失败原因 |
| 8.4.5.2 模型推理——基于LangChain框架实现ReActAgent | [`8.4、Agent/8.4.5 Agent实战/8.4.5.2 模型推理——基于LangChain框架实现ReActAgent/8.4.5.2_ReActAgent检索问答.py`](<8.4、Agent/8.4.5 Agent实战/8.4.5.2 模型推理——基于LangChain框架实现ReActAgent/8.4.5.2_ReActAgent检索问答.py>) | 演示ReAct智能体 |
| 8.4.5.3 模型推理——基于LangChain框架实现KG-RAG-Agent | [`8.4、Agent/8.4.5 Agent实战/8.4.5.3 模型推理——基于LangChain框架实现KG-RAG-Agent/8.4.5.3_知识图谱RAG智能体.py`](<8.4、Agent/8.4.5 Agent实战/8.4.5.3 模型推理——基于LangChain框架实现KG-RAG-Agent/8.4.5.3_知识图谱RAG智能体.py>) | 演示知识图谱RAG智能体 |
| 8.4.5.4 模型部署FastAPI——基于LangChain框架和FastAPI实现部署ToolAgent服务 | [`8.4、Agent/8.4.5 Agent实战/8.4.5.4 模型部署FastAPI——基于LangChain框架和FastAPI实现部署ToolAgent服务/8.4.5.4_ToolAgent_FastAPI服务.py`](<8.4、Agent/8.4.5 Agent实战/8.4.5.4 模型部署FastAPI——基于LangChain框架和FastAPI实现部署ToolAgent服务/8.4.5.4_ToolAgent_FastAPI服务.py>) | 演示ToolAgent服务部署 |
| 8.5.2 长上下文建模实战案例 | [`8.5 长上下文建模/8.5.2 长上下文建模实战案例/8.5.2_合并PEFT适配器.py`](<8.5 长上下文建模/8.5.2 长上下文建模实战案例/8.5.2_合并PEFT适配器.py>) | 将兼容的适配器合并到基础模型；合并操作本身不扩大上下文窗口 |

## 建议阅读顺序

先阅读 8.2 的分块与基础 RAG，再看多轮和混合检索；随后阅读 8.3 的工具调用、8.4 的 Agent，最后阅读 8.5 的适配器合并。

## 注意事项

- RAG 需要相应语料、模型或服务；文件中的外部项目操作说明不代表配书目录包含完整框架。
- 本地 PDF 混合检索使用中文分词进行 BM25 检索；归一化向量的 L2 排序与余弦相似度排序等价。
- 工具模式描述参数，工具执行由应用程序完成；模拟天气和航班数据不用于真实查询。
- 搜索服务比较的评分来自模型辅助判断，需人工核验来源与事实；服务失败单独记录。
- 知识图谱案例会导入数据并执行生成的 Cypher，应使用专用教学数据库。
- 适配器须与基础模型兼容；长上下文能力还依赖模型配置与适配器训练，不能由权重合并单独获得。
<!-- CODE_OVERVIEW:END -->
