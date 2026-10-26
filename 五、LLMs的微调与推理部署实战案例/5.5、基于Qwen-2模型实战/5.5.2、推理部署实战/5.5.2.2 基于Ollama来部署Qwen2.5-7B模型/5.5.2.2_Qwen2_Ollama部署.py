# 文件用途：说明Ollama部署Qwen模型
# 对应章节/小节：第5章 5.5.2.2 基于Ollama来部署Qwen2.5-7B模型
# 作者：NYY
# 创建日期：2024-10-13（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# 环境准备
# # 0.1、环境配置
# >> 本机硬件
# 显存：
#
# >> 安装依赖
#
# >> 安装Ollama。
# 下载地址：https://github.com/ollama/ollama



# 完整代码
# Step1、启动 Ollama 服务
# 安装 ollama 后，您可以使用以下命令启动 ollama 服务：
# ollama serve  # 使用 ollama 时需要保持此服务运行


# Step2、拉取并测试 Qwen2.5 模型
# 要拉取模型检查点并运行模型，请使用 ollama run 命令。您可以通过添加后缀到 qwen2.5 来指定模型大小，例如 :0.5b, :1.5b, :7b, 或 :72b：
# ollama run qwen2.5:7b
# 要退出，请输入“/bye”并按回车键。


# Step3、通过兼容 OpenAI 的接口请求模型
# 您还可以通过其兼容 OpenAI 的 API 访问 ollama 服务。请注意，您需要 (1) 在使用 API 时保持 ollama serve 运行，(2) 在使用此 API 之前执行 ollama run qwen2.5:7b 以确保模型检查点已准备好。
from openai import OpenAI
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',  # OpenAI 客户端要求填写此参数；本地 Ollama 服务不使用该值进行鉴权
)
chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'Say this is a test',
        }
    ],
    model='qwen2.5:7b',
)
print(chat_completion.choices[0].message.content or "")
