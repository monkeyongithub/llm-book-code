# 文件用途：演示GLM模型推理
# 对应章节/小节：第1章 1.3.4 GLM家族
# 作者：NYY
# 创建日期：2024-10-22（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：模型家族介绍的概念扩展示例

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

# Step1、加载 GLM-4 分词器
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)

# Step2、按聊天模板编码用户问题
query = "你好"

inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

inputs = inputs.to(device)
# Step3、加载模型并生成回复
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/glm-4-9b-chat",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
