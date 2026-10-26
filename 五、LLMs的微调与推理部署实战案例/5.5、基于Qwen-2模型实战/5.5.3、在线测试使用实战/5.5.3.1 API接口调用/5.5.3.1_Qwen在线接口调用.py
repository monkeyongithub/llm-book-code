# 文件用途：演示Qwen在线API调用
# 对应章节/小节：第5章 5.5.3.1 API接口调用
# 作者：NYY
# 创建日期：2024-10-13（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# API接口调用
# 使用 qwen-plus 演示在线文本生成，其他模型可调整 model 参数。
import os
from openai import OpenAI

# Step1、配置 Qwen 在线接口客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# Step2、发送聊天消息
completion = client.chat.completions.create(
    model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '你是谁？'}],
)

# Step3、展示接口返回结果
print(completion.model_dump_json())
