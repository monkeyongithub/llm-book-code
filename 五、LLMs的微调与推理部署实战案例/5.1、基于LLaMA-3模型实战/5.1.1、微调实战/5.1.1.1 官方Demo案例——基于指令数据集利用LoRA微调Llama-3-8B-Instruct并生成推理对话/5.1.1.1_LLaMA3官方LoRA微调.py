# 文件用途：演示LLaMA-3指令微调
# 对应章节/小节：第5章 5.1.1.1 官方Demo案例——基于指令数据集利用LoRA微调Llama-3-8B-Instruct并生成推理对话
# 作者：NYY
# 创建日期：2024-10-11（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

'''
参考原文：https://github.com/datawhalechina/self-llm/blob/master/models/LLaMA3/LLaMA3-8B-Instruct%20Lora.ipynb
解读：https://github.com/datawhalechina/self-llm/blob/master/LLaMA3/04-LLaMA3-8B-Instruct%20Lora%20%E5%BE%AE%E8%B0%83.md
LLMs之LLaMA3：基于指令数据集利用LoRA微调Llama-3-8B-Instruct并进行推理对话生成——定义数据集(JSON文件,instruction-input-output格式)→数据预处理(对每一个输入样本执行分词和截断等操作+加载分词器并解码部分样例输出)
→模型微调(加载模型并启用BF16和输入的梯度计算+配置LoRA参数并更新到LLMs模型然后输出可训练参数的数量+配置训练参数【梯度累积步技术+梯度检查点技术】+模型训练微调+保存模型及其分词器)→
模型推理(加载tokenizer并将基础模型转换为带有LoRA微调的模型+构建聊天模板列表并使用tokenizer将其转换为适合聊天生成的文本格式+使用tokenizer将文本转换为张量并其移至CUDA设备+使用tokenize对文本分词并转换为张量并其移至CUDA设备+文本生成并解码ID列表为字符串文本)
'''

# 环境准备
# 环境配置
# >> 本机硬件
# 显卡 3090-24G，ubuntu系统 22.04，Python版本 3.10，Cuda版本 12.1，PyTorch版本 2.1.0


# >> 安装依赖
# 注意：推荐安装Anaconda，以管理Python环境和依赖库
# pip install modelscope==1.9.5
# pip install "transformers>=4.40.0"
# pip install streamlit==1.24.0
# pip install sentencepiece==0.1.99
# pip install accelerate==0.29.3
# pip install datasets==2.19.0
# pip install peft==0.10.0
# MAX_JOBS=8 pip install flash-attn --no-build-isolation
# 注意：flash-attn 安装会比较慢，大概需要十几分钟。

# 文件下载
# >> 下载数据集
# 下载地址：https://github.com/datawhalechina/self-llm/blob/master/dataset/huanhuan.json

# >> 下载模型：模型大小为 15 GB，下载模型大概需要 2 分钟
# import torch
# from modelscope import snapshot_download, AutoModel, AutoTokenizer
# import os
# model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', cache_dir='models', revision='master')


# 完整代码
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig

# process_func函数：处理输入数据，生成模型所需的输入ID、注意力掩码和标签
def process_func(example):
    MAX_LENGTH = 384  # Llama分词器会将一个中文字切分为多个token，因此设置合适长度以保证数据完整性
    input_ids, attention_mask, labels = [], [], []
    # 使用 tokenizer 处理用户指令和输入
    messages = [{"role": "user", "content": example["instruction"] + example["input"]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    instruction = tokenizer(prompt, add_special_tokens=False)
    response = tokenizer(example["output"] + "<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }



# Step1、读取指令数据集
df = pd.read_json('./huanhuan.json')
ds = Dataset.from_pandas(df)
print(ds[:3])


# Step2、将样本编码为训练输入与标签
# 利用process_func函数处理数据集中的每个示例：对每一个输入样本执行分词和截断，并生成模型所需的input_ids、attention_mask和labels
tokenizer = AutoTokenizer.from_pretrained('models/LLM-Research/Meta-Llama-3-8B-Instruct', use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
# 过长提示被截断后可能没有任何目标 token，这类样本不能用于计算训练损失。
tokenized_id = tokenized_id.filter(lambda sample: any(label != -100 for label in sample["labels"]))
if not len(tokenized_id):
    raise ValueError("截断后没有有效训练目标，请缩短提示或增大 MAX_LENGTH")
print(tokenized_id)

# 加载分词器并解码部分样例输出
# 加载预训练的Llama分词器，将填充标记设置为eos标记，并打印填充标记及其ID，同时解码并输出指定的input_ids和labels
print(tokenizer.pad_token, tokenizer.pad_token_id, tokenizer.eos_token_id)
print(tokenizer.decode(tokenized_id[0]['input_ids']))
print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[0]["labels"]))))


# Step3、配置 LoRA 并微调模型
# 加载模型并启用BF16和输入的梯度计算
# 加载预训练的因果语言模型，自动分配设备并设置数据类型为bfloat16，同时启用输入的梯度计算，并打印模型的数据类型。
import torch
model = AutoModelForCausalLM.from_pretrained('models/LLM-Research/Meta-Llama-3-8B-Instruct',
                                             device_map="auto" ,torch_dtype=torch.bfloat16)
model.enable_input_require_grads()
model.config.use_cache = False
print(model.dtype)


# 配置LoRA参数并更新到LLMs模型然后输出可训练参数的数量：包括设置因果语言模型任务类型、训练目标模块、秩、缩放因子和dropout比例
from peft import LoraConfig, TaskType, get_peft_model
# 配置LoRA参数
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    # 设置模型要训练的层，主要就是attention部分的层
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
print(config)
# 获取配置后的可训练模型并打印模型中可训练参数的数量
model = get_peft_model(model, config)
model.print_trainable_parameters()


# 配置训练参数：包括输出目录、每个设备的训练批次大小、梯度累积步数、日志记录步数、训练轮数、保存步数、学习率、每个节点保存模型以及启用梯度检查点
args = TrainingArguments(
    output_dir="./output/llama3",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # 如果显存比较小，可以把 batch_size 设置小一点，梯度累加增大一些。
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)


# 模型训练微调：实例化训练器同时配置模型、训练参数、数据集并启动模型训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()


# 保存模型：保存微调训练后的模型及其分词器，用于后续的推理任务
peft_model_id ="./llama3_lora"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)



# Step4、加载适配器进行对话推理
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

# 加载tokenizer并将基础模型转换为带有LoRA微调的模型
# 释放训练器引用后再加载，避免两份基础模型同时占用显存。
import gc
del trainer, model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
mode_path = 'models/LLM-Research/Meta-Llama-3-8B-Instruct' 
lora_path = './llama3_lora'
tokenizer = AutoTokenizer.from_pretrained(mode_path)
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto" ,torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, model_id=lora_path).eval()

# 构建聊天模板列表并使用tokenizer将其转换为适合聊天生成的文本格式
instruct =  "现在你要扮演皇帝身边的女人--甄嬛"
user_input = "你是谁？"
messages = [
    {"role": "system", "content": instruct},
    {"role": "user", "content": user_input}
]
# messages是一个包含对话信息的列表，tokenize=False表示不立即对文本进行分词，add_generation_prompt=True表示添加一个提示符，指示模型开始生成文本。
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# 聊天模板已含特殊标记，编码时不重复添加，再将张量放到模型输入设备。
model_inputs = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(model.device)

# 文本生成并解码ID列表为字符串文本
# 使用模型生成文本，提取生成部分的ID并解码为字符串以获取最终响应
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
    eos_token_id=tokenizer.convert_tokens_to_ids('<|eot_id|>')
)
# 提取生成的ID，移除了输入部分的ID，只保留新生成部分
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
# 解码生成的ID为字符串文本，跳过特殊token
# batch_decode方法可以同时处理多个序列的解码。skip_special_tokens=True表示在解码过程中忽略特殊的令牌。最后，通过索引[0]获取第一个解码结果，即生成的回复
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response) # 打印生成的响应
