# 文件用途：演示LLaMA模型推理
# 对应章节/小节：第1章 1.3.2 LLaMA家族
# 作者：NYY
# 创建日期：2024-10-22（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：模型家族介绍的概念扩展示例

import transformers
import torch
torch.cuda.empty_cache()


# Step1、加载 LLaMA-3 指令模型
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
  "text-generation",
  model=model_id,
  model_kwargs={"torch_dtype": torch.bfloat16},
  device_map="auto",
)

# Step2、把系统指令和用户问题组成聊天提示
instruction = "You are a pirate chatbot who always responds in pirate speak!"
user_input = "Who are you?"

messages = [
  {"role": "system", "content": instruction},
  {"role": "user", "content": user_input},
]

prompt = pipeline.tokenizer.apply_chat_template(
  messages,
  tokenize=False,
  add_generation_prompt=True
)

# Step3、配置终止标记并生成回复
terminators = [
  pipeline.tokenizer.eos_token_id,
  pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
  prompt,
  max_new_tokens=256,
  eos_token_id=terminators,
  do_sample=True,
  temperature=0.6,
  top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])
