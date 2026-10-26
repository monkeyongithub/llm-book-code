# 文件用途：演示Qwen模型推理
# 对应章节/小节：第5章 5.5.2.1 基于Transformers框架实现测试Qwen2.5-7B-Instruct模型
# 作者：NYY
# 创建日期：2024-10-13（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# 本节示例：基于Transformers框架实现测试Qwen2.5-7B-Instruct模型
# 源码地址：https://github.com/QwenLM/Qwen2.5
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step1、加载 Qwen2.5 模型和分词器
model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step2、按聊天模板编码问题
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(model.device)

# Step3、生成并解码回复
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
