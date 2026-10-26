# 文件用途：说明知识库Web部署
# 对应章节/小节：第8章 8.2.6.2 基于LangChain-ChatGLM框架实现RAG
# 作者：NYY
# 创建日期：2024-10-29（文件系统记录）
# 运行方式：阅读示例说明
# 备注：操作说明，不包含独立执行逻辑

# Step1、准备 WebUI 项目
# # (1)、软硬件要求
# # 项目需要Python>=3.8.1, 默认已安装torch
#
# # (2)、安装项目
# 在终端中执行 git clone，克隆项目：
# git clone https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui.git
# # 若国内用户访问Github存在网络问题, 也可以执行:
# git clone https://openi.pcl.ac.cn/Learning-Develop-Union/LangChain-ChatGLM-Webui.git
#
# # (3)、安装依赖包
# cd LangChain-ChatGLM-Webui       # 进入本项目目录
# pip install -r requirements.txt        # 国内用户可设置清华源加速下载. 
# 另: 若您想要安装测试ModelScope版本, 需要额外安装ModelScope包: 
# pip install modelscope==1.4.3 -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html 
# OpenI启智社区部署推荐的Docker镜像:
#
# dockerhub.pcl.ac.cn:5000/user-images/openi:LangChain-ChatLLM-2.0


# Step2、选择 Python 或 Docker 方式启动
# T1、Python 启动：基于 Python脚本直接启动
# 示例：Hugging Face 版本
# 在terminal中执行命令:
# python3 app.py

# 示例：ModelScope 版本
# 进入modelscope文件目录：
# cd modelscope
# python3 app.py

# T2、Docker 启动：基于 Docker环境启动
# (1)、运行镜像
# docker run -it --rm --runtime=nvidia --gpus all --network host registry.cn-beijing.aliyuncs.com/public-development-resources/langchain-chatglm-webui:Base bash

# (2)、git clone项目
# git clone https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui.git

# (3)、安装依赖包
# cd LangChain-ChatGLM-Webui
# pip3 install -r requirements.txt

# (4)、启动
# python3 app.py
