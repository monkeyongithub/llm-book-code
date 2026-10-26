# 文件用途：说明Ollama与Dify RAG部署
# 对应章节/小节：第8章 8.2.6.6 基于OLLaMA后端+Dify前端框架实现RAG
# 作者：NYY
# 创建日期：2024-10-30（文件系统记录）
# 运行方式：阅读示例说明
# 备注：操作说明，不包含独立执行逻辑

# 基于Ollama后端框架(配置phi3/LLaMA-3模型)结合Dify前端框架(设置知识库文件+向量化存储+应用发布)创建包括实现本地知识库问答/翻译助手等多个应用



# Step1、准备 Docker、Ollama 与 Dify
# 安装Docker和Docker Compose，

# 安装并配置Ollama
# 安装并配置Ollama

# 打开Dify界面并配置模型
'''
安装并启动Dify
打开界面
浏览器中访问http://localhost/install并开始初始化过程，设置账号，之后用此账号登录。
'''

# Step2、配置模型、知识库并发布问答应用
# 配置模型
# 模型名称：phi3:latest
# 容器地址：http://host.docker.internal:11434






# 创建知识库并向量化存储
# 点击知识库→创建知识库→添加PDF资料→保存并处理





# 创建应用并发布
# 创建应用→设置角色→添加知识库资料→并设置提示词→选择大语言模型→依次点击更新→发布→Start Chat即可！


# 基于知识库进行对话测试
