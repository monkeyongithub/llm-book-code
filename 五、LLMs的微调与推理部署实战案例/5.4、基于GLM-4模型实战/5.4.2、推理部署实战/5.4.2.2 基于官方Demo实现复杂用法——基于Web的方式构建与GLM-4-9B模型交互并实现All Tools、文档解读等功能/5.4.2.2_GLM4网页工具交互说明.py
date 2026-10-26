# 文件用途：演示GLM-4网页和工具交互
# 对应章节/小节：第5章 5.4.2.2 基于官方Demo实现复杂用法——基于Web的方式构建与GLM-4-9B模型交互并实现All Tools、文档解读等功能
# 作者：NYY
# 创建日期：2024-10-12（文件系统记录）
# 运行方式：阅读示例说明
# 备注：操作说明，不包含独立执行逻辑

# 本节示例：复杂用法—基于Web的方式构建与GLM-4-9B模型交互并实现All Tools、文档解读等功能
'''
源码地址：https://github.com/THUDM/GLM-4/tree/main/composite_demo
'''

# 环境准备
# 环境配置
# >> 本机硬件
# 显存： Python 3.10

# >> 安装依赖
# 注意：推荐安装Anaconda，以管理Python环境和依赖库
# 新建一个 conda 环境并安装所需依赖：
# conda create -n glm-4-demo python=3.12
# conda activate glm-4-demo
# pip install -r requirements.txt

# 若要使用 Code Interpreter 还需要安装 Jupyter 内核：
# ipython kernel install --name glm-4-demo --user
# 注意：若在使用 All Tools 的 Python 代码执行能力时使用 Matplotlib 画图，可以在 argv 数组中添加 "--matplotlib=inline"


# 若要使用浏览器和搜索功能，还需要启动浏览器后端。首先，根据 Node.js 官网的指示安装 Node.js，然后安装包管理器 PNPM 之后安装浏览器服务的依赖：
# cd browser
# npm install -g pnpm
# pnpm install
# 注意，需要修改 Bing 搜索的API Key ，在browser/src/config.ts 中的 BING_SEARCH_API_KEY 文件内


# 文件下载
# >> 下载本仓库：
# git clone https://github.com/THUDM/GLM-4
# cd GLM-4

# >> 下载数据集


# >> 下载模型
# 模型地址：https://huggingface.co/THUDM/glm-4-9b




# 完整代码
# Step1、启动浏览器后端
# cd browser
# pnpm start

# Step2、加载 GLM-4 模型并启动网页示例
# streamlit run src/main.py
# 注意：初次访问需要下载并加载模型，可能需要花费一定时间。如果已经在本地下载了模型，可以通过 将对应的 *_MODEL_PATH 设为 models/glm-4-9b-chat 等根目录相对路径来加载模型，THUDM/glm-4-9b-chat
# Chat 模型支持使用 vLLM 推理。若要使用，请安装 vLLM 并设置环境变量 USE_VLLM=1。


# Step3、按书稿示例体验工具与文档模式
# 对话模式: 用户可以直接在侧边栏修改 top_p, temperature 等参数来调整模型的行为。
# (1)、All Tools: 具有完整工具调用能力的对话模式，原生支持网页浏览、代码执行、图片生成，并支持自定义工具。注意本模式下不支持系统提示词，模型会自动构建提示词。
# >> 模型拥有进行网页搜索和 Python 代码执行的能力。同时，模型也可以连续调用多个工具。比如模型通过调用浏览器工具进行搜索获取到了需要的数据，之后将会调用 Python 工具执行代码，利用 Matplotlib 绘图。
# (2)、文档解读: 支持上传文档进行文档解读与对话。注意本模式下不支持工具调用和系统提示词。
# >> 用户可以上传文档，使用 GLM-4-9B的长文本能力，对文本进行理解。可以解析 pptx，docx，pdf等文件。
