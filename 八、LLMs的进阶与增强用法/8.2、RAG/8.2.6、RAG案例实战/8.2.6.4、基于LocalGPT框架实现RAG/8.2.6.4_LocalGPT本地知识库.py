# 文件用途：说明LocalGPT RAG部署
# 对应章节/小节：第8章 8.2.6.4 基于LocalGPT框架实现RAG
# 作者：NYY
# 创建日期：2024-10-30（文件系统记录）
# 运行方式：阅读示例说明
# 备注：操作说明，不包含独立执行逻辑

# Step1、准备 LocalGPT 项目和环境
# # (1)、软硬件要求
# Python版本，Python 3.10及其以上
# C++编译器，如果在 pip安装过程中构建 wheel 时遇到错误，可能需要在你的计算机上安装 C++ 编译器。
#
# # (2)、安装项目
# git clone https://github.com/PromtEngineer/localGPT.git
# cd localGPT
#
# # 为虚拟环境管理安装 conda。创建并激活一个新的虚拟环境。
# conda create -n localGPT python=3.10.0
# conda activate localGPT
#
#
# # (3)、安装依赖
# pip install -r requirements.txt
#
#
# # (4)、安装 LLAMA-CPP：
# LocalGPT 使用 LlamaCpp-Python 来支持 GGML（你需要 llama-cpp-python <=0.1.76）和 GGUF（llama-cpp-python >=0.1.83）模型。
# 若要运行量化后的 Llama3 模型，请确保已安装版本 0.2.62 或更高版本的 llama-cpp-python。
# 如果你想使用 BLAS 或 Metal 与 llama-cpp，你可以设置相应的标志：
# 对于 NVIDIA GPU 支持，使用 cuBLAS
# 示例：cuBLAS
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir


# Step2、配置文档与本地模型
# (1)、下载测试数据集
# 为了测试，这个仓库包含了一份美国宪法文件作为示例文件供你使用。
# 下载地址：https://constitutioncenter.org/media/files/constitution.pdf

# (2)、导入你自己的数据
# 将你的文件放入 SOURCE_DOCUMENTS 文件夹中。你可以在 SOURCE_DOCUMENTS 文件夹内放置多个文件夹，代码会递归地读取你的文件。

# (3)、支持的文件格式：
# LocalGPT 当前支持以下文件格式。LocalGPT 使用 LangChain 加载这些文件格式。在 constants.py 中的代码使用 DOCUMENT_MAP 字典来映射文件格式到对应的加载器。为了添加对另一种文件格式的支持，只需在这个字典中添加文件格式及其对应来自 LangChain 的加载器即可。
# DOCUMENT_MAP = {
#     ".txt": TextLoader,
#     ".md": TextLoader,
#     ".py": TextLoader,
#     ".pdf": PDFMinerLoader,
#     ".csv": CSVLoader,
#     ".xls": UnstructuredExcelLoader,
#     ".xlsx": UnstructuredExcelLoader,
#     ".docx": Docx2txtLoader,
#     ".doc": Docx2txtLoader,
# }

# (4)、导入
# 运行以下命令以导入所有数据。
# T1、如果你的系统上设置了 cuda。
# python ingest.py
# 你会看到类似这样的输出

# T2、使用设备类型参数来指定给定设备。要在 CPU 上运行
# python ingest.py --device_type cpu
# 这将创建一个名为 DB 的新文件夹，并将其用于新创建的向量存储。你可以根据需要导入任意数量的文档，所有文档都会累积到本地嵌入数据库中。如果你想从空数据库开始，可以删除 DB 并重新导入文档。
# 注意：当你第一次运行时，它需要互联网连接来下载嵌入模型（默认：Instructor Embedding）。后续运行中，不会有任何数据离开你的本地环境，你可以无需互联网连接进行数据导入。
# ingest.py代码解读—加载文档并分割成块生成嵌入获取VDB的脚本：配置日志记录→解析Click命令→加载文档并分为两类(基于文件扩展名将文档分类为text文档或Python代码文档)→文本分割并存储(均采用字符递归分割)→生成嵌入(嵌入模型默认instructor-large)→创建和持久化向量数据库(基于Chroma)


# Step3、导入文档并发起本地问答
# (1)、为了与你的文档聊天，运行以下命令
# 默认情况下在 cuda 上运行
# python run_localGPT.py
# 这将加载已导入的向量存储和嵌入模型。

# 生成答案后，你可以继续询问另一个问题而不需要重新运行脚本，只需等待再次出现提示即可。
# 注意：当你首次运行时，它需要互联网连接来下载大语言模型（默认：TheBloke/Llama-2-7b-Chat-GGUF）。之后，即使关闭互联网连接，脚本推断仍然可以工作。没有数据会离开你的本地环境。
# 输入 exit 以结束脚本。

# run_localGPT.py代码解读—执行信息检索任务的脚本：配置日志记录→解析Click命令→加载嵌入模型(默认instructor-large)→加载由inget.py创建的VDB向量数据库(基于Chroma)并创建检索器→加载本地LLM(默认llama-3-8b或Mistral-7B-Instruct)→设置问答检索链→创建RetrievalQA对象(执行检索和问题回答任务)→与用户互动问答(基于用户输入获取查询问题+RQA对象获取问题答案和源文档)
