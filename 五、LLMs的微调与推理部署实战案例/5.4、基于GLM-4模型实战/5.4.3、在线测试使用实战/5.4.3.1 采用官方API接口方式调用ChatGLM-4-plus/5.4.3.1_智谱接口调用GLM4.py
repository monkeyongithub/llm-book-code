# -*- coding: utf-8 -*-
# 文件用途：演示GLM-4在线API调用
# 对应章节/小节：第5章 5.4.3.1 采用官方API接口方式调用ChatGLM-4-plus
# 作者：NYY
# 创建日期：2024-10-13（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# 采用官方API接口方式调用
# 源码地址：https://bigmodel.cn/console/trialcenter?modeCode=glm-4-plus
# pip install zhipuai 请先在终端进行安装
import os

api_key = os.environ["ZHIPUAI_API_KEY"]

from zhipuai import ZhipuAI

# Step1、配置 GLM-4 客户端
client = ZhipuAI(api_key=api_key)
# Step2、发送包含工具说明的对话请求
response = client.chat.completions.create(
    model="glm-4-plus",
    messages=[
        {
            "role": "system",
            "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"
        },
        {
            "role": "user",
            "content": "你好"
        }
    ],
    top_p= 0.7,
    temperature= 0.95,
    max_tokens=1024,
    tools = [{"type":"function","function":{"name":"get_weather","description":"获取指定地点当前的天气情况","parameters":{"type":"object","properties":{"location":{"type":"string","description":"城市，例如：北京、上海"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}],
    stream=True
)
# Step3、查看流式返回内容
for trunk in response:
    print(trunk)
