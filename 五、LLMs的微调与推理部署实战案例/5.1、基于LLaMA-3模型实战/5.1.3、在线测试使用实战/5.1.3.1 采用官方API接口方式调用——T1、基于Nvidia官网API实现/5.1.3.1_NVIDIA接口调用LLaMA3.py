# 文件用途：演示LLaMA-3在线API调用
# 对应章节/小节：第5章 5.1.3.1 采用官方API接口方式调用——T1、基于Nvidia官网API实现
# 作者：NYY
# 创建日期：2024-10-10（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# 采用API接口方式调用


# 本节示例：基于Nvidia官网API实现
import os

from openai import OpenAI

# Step1、配置 NVIDIA 兼容接口客户端
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.environ["NVIDIA_API_KEY"]
)

# Step2、发送 LLaMA-3 对话请求
completion = client.chat.completions.create(
  model="meta/llama3-70b-instruct",
  messages=[{"role":"user","content":"Write a limerick about the wonders of GPU computing."}],
  temperature=0.5,
  top_p=1,
  max_tokens=1024,
  stream=True
)

# Step3、逐块展示流式回复
for chunk in completion:
  if chunk.choices and chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")
