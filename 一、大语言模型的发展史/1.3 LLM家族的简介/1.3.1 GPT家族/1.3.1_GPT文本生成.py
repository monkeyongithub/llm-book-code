# 文件用途：演示GPT-2生成与GPT API调用
# 对应章节/小节：第1章 1.3.1 GPT家族
# 作者：NYY
# 创建日期：2024-10-22（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：模型家族介绍的概念扩展示例

# T1、使用 Transformers 与 GPT-2 演示文本续写
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载预训练的GPT-2模型和分词器
# Step1、加载 GPT-2 本地模型与分词器
model_name = 'gpt2'  # 你可以选择其他版本的GPT模型，如'gpt2-medium', 'gpt2-large', 'gpt2-xl'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 编码输入文本
# Step2、编码文本提示
input_text = "Hello, my dog is"
encoded_input = tokenizer.encode(input_text, return_tensors='pt')

# 生成文本
# Step3、生成并解码文本
output = model.generate(encoded_input, max_length=50, num_return_sequences=1)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)



# T2、利用OpenAI的API接口采用GPT-4系列模型实现对话生成
# 来源地址：https://platform.openai.com/docs/quickstart?lang=ChatCompletions

from openai import OpenAI

# OpenAI 客户端从环境变量读取密钥。

# 请求ChatGPT生成对话
# Step4、创建在线 API 客户端
client = OpenAI()
# Step5、提交消息并展示返回结果
completion = client.chat.completions.create(
  model="gpt-4o-mini",   # gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)
print(completion.choices[0].message)
