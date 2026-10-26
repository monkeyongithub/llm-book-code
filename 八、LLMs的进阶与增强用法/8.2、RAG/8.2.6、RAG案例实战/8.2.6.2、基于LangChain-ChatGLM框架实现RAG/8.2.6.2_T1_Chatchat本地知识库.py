# 文件用途：说明本地知识库部署
# 对应章节/小节：第8章 8.2.6.2 基于LangChain-ChatGLM框架实现RAG
# 作者：NYY
# 创建日期：2024-10-29（文件系统记录）
# 运行方式：阅读示例说明
# 备注：操作说明，不包含独立执行逻辑

# Step1、准备 Chatchat 项目
# # (1)、软硬件要求
# 软件方面，本项目已支持在 Python 3.8-3.11 环境中进行使用，并已在 Windows、macOS、Linux 操作系统中进行测试。
# 硬件方面，因 0.3.0 版本已修改为支持不同模型部署框架接入，因此可在 CPU、GPU、NPU、MPS 等不同硬件条件下使用。
#
# # (2)、安装项目
# 注意：为避免依赖冲突，请将 Langchain-Chatchat 和模型部署框架如 Xinference 等放在不同的 Python 虚拟环境中, 比如 conda, venv, virtualenv 等。
# 安装 Langchain-Chatchat
# pip install langchain-chatchat -U
#
#
# # (3)、安装依赖
# 因模型部署框架 Xinference 接入 Langchain-Chatchat 时需要额外安装对应的 Python 依赖库，因此如需搭配 Xinference 框架使用时，建议使用如下安装方式：
# pip install "langchain-chatchat[xinference]" -U



# Step2、配置模型与数据目录
# 设置 Chatchat 存储配置文件和数据文件的根目录（可选）
# 在 Linux 或 macOS 终端中设置
# export CHATCHAT_ROOT=external/chatchat_data
# 在 Windows 命令提示符中设置
# set CHATCHAT_ROOT=external/chatchat_data
# 若不设置该环境变量，则自动使用当前目录。

# 执行初始化：创建所有需要的数据目录，复制 samples 知识库内容，生成默认 yaml 配置文件
# chatchat init

# 修改配置文件
# 配置模型（model_settings.yaml）
# 默认选用的 LLM 名称
# DEFAULT_LLM_MODEL: qwen1.5-chat

# 默认选用的 Embedding 名称
# DEFAULT_EMBEDDING_MODEL: bge-large-zh-v1.5
# 将 `LLM_MODEL_CONFIG` 中 `llm_model, action_model` 的键改成对应的 LLM 模型
# 在 `MODEL_PLATFORMS` 中修改对应模型平台信息

# 配置知识库路径（basic_settings.yaml）
# 知识库默认存储路径
# KB_ROOT_PATH: data/knowledge_base

# 数据库默认存储路径。如果使用sqlite，可以直接修改DB_ROOT_PATH；如果使用其它数据库，请直接修改SQLALCHEMY_DATABASE_URI。
# DB_ROOT_PATH: data/knowledge_base/info.db

# 知识库信息数据库连接URI
# SQLALCHEMY_DATABASE_URI: sqlite:///data/knowledge_base/info.db

# Step3、初始化知识库
# 进行知识库初始化前，请确保已经启动模型推理框架及对应 embedding 模型，且已按照上述步骤3完成模型接入配置。
# chatchat kb -r

# 若输出出现以下日志即为成功:
# ----------------------------------------------------------------------------------------------------
# 知识库名称      ：samples
# 知识库类型      ：faiss
# 向量模型：      ：bge-large-zh-v1.5
# 知识库路径      ：data/knowledge_base/samples
# 文件总数量      ：47
# 入库文件数      ：42
# 知识条目数      ：740
# 用时            ：0:02:29.701002
# ----------------------------------------------------------------------------------------------------
# 总计用时        ：0:02:33.414425


# Step4、启动服务并测试问答
# 出现新界面即为启动成功
# chatchat start -a
