# 文件用途：演示GPT-4o API调用
# 对应章节/小节：第5章 5.3.1 基于官方API在线推理测试
# 作者：NYY
# 创建日期：2024-10-11（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# 环境准备
# 源码地址：https://platform.openai.com/docs/quickstart?desktop-os=windows&quickstart-example=completions&language-preference=python
# # 0.1、环境配置
# >> 本机硬件
# 注册账号获取密钥
# 地址：https://platform.openai.com/docs/api-reference/authentication
#
#
# >> 安装依赖
# pip install openai



# 完整代码
# setx OPENAI_API_KEY "your_api_key_here"

from openai import OpenAI
# Step1、创建 GPT-4o API 客户端
client = OpenAI()

# Step2、发送聊天消息
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)
# Step3、展示模型返回内容
print(completion.choices[0].message)
