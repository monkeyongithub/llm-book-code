# 文件用途：说明Ollama与AnythingLLM RAG部署
# 对应章节/小节：第8章 8.2.6.5 基于OLLaMA+AnythingLLM框架实现RAG
# 作者：NYY
# 创建日期：2024-10-30（文件系统记录）
# 运行方式：阅读示例说明
# 备注：操作说明，不包含独立执行逻辑

# Step1、安装并启动 Ollama

# Step2、安装 AnythingLLM

# Step3、连接 Ollama 与工作区
# ollama开启服务
# 一般会默认开启该服务，只需访问验证即可！
# >> 通过终端执行命令行，
# ollama serve
#
# >> 访问http://localhost:11434 进行验证，返回以下内容，即正确！Ollama is running
# curl http://localhost:11434
# >> 列出下载的模型
# ollama list
#
# 配置AngthingLLM
# 界面初始打开时，会一路设置，后期可以根据下述切换栏进行对应设置
# >> LLM Preference：LLM provider选择ollama，依次填写ollama的服务URL地址http://127.0.0.1:11434，下载的模型，token(默认4096)；
# >> Embedding Preferenc：同样选择ollama，其余基本一致，max我看默认8192，我也填了8192；
# >> Vector Database：选择默认的LanceDB即可，也可以选用Chroma、Milvus、Pinecone等向量数据库。

# Step4、测试普通对话和文档问答
# AnythingLLM引入工作区（workspace）的概念，作为文档的容器，可以在一个工作区内共享文档，但是工作区之间隔离。
# T1、新建工作区(workspace)并测试Chat功能
# T2、新建工作区(workspace)并测试RAG功能：上传文档并在界面中实现对话查询
# 简介
# AnythingLLM既可以上传文档，也可以抓取网页信息。
# 实现思路：比如先填写网页地址，点击Fetch抓取网页资料，然后勾选对应的文档资料，保存到对应工作区中的向量数据库中。
