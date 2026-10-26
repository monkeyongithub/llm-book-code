# 文件用途：演示unsloth微调
# 对应章节/小节：第5章 5.1.1.4 基于Colab平台利用unsloth优化框架实现LoRA微调llama-3-8b
# 作者：NYY
# 创建日期：2024-10-11（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# LLMs之LLaMA-3：基于Colab平台(采用T4 GPU+至少37G)基于ruozhiba_LLaMA3中文语料数据利用unsloth优化框架(速度更快/量化功能)并采用LoRA技术微调llama-3-8b-bnb(合并原始模型和LoRA模型)同时进行格式转换和4位量化最后模型导出——加载量化模型并推理测试原始模型效果(使用Unsloth库以4-bit精度加载llama-3-8b模型+基于alpaca_prompt模板构建指令和输入的提示测试加载模型的响应(流式输出))→定义数据集(准备并格式化微调数据集为alpaca_prompt)→模型微调(配置LoRA参数和训练参数【采用梯度检查点+梯度累积+8位AdamW优化器+权重衰减+学习率调度器等技术】→开始训练→测试微调后的模型→保存LoRA模型→合并LoRA模型并量化和转换格式【16位的hf格式→16位的gguf格式→4位的gguf格式】)→下载量化后的模型到本地
'''
部分参考：https://colab.research.google.com/drive/1MfP3vt9YrOkjg70dKPPFvB174PBnaPSB?usp=sharing#scrollTo=x8QigZLgRRHM
'''

# 环境准备
# 环境配置
# >> 本机硬件
# 本示例在 Colab 平台演示。


# >> 安装依赖
# 注意：推荐安装Anaconda，以管理Python环境和依赖库
# 微调库unsloth
# %%capture
# import torch
# major_version, minor_version = torch.cuda.get_device_capability()
# 由于Colab有torch 2.2.1，会破坏软件包，要单独安装
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# if major_version >= 8:
    # 新GPU，如Ampere、Hopper GPU（RTX 30xx、RTX 40xx、A100、H100、L40）。
#     !pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
# else:
    # 较旧的GPU（V100、Tesla T4、RTX 20xx）
#     !pip install --no-deps trl peft accelerate bitsandbytes
#     !pip install xformers==0.0.25  #最新的0.0.26不兼容
# pass

# 完整代码
# Step1、加载量化模型并观察微调前输出
# 加载模型：使用Unsloth库以4-bit精度加载llama-3-8b模型并定最大序列长度为2048
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",  # 源自HuggingFace官方路径
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


# 微调前测试：基于alpaca_prompt模板构建指令和输入的提示测试加载模型的响应(流式输出)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""
FastLanguageModel.for_inference(model)
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "请用中文回答",  # instruction：用户指令
            "海绵宝宝的书法是不是叫做海绵体？",  # input：补充输入
            "",  # output：模型回答
        )
    ], return_tensors="pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)




# Step2、加载并格式化指令数据
# 从数据集中加载训练数据，并通过格式化函数生成带有结束标记的文本，以确保模型正确处理指令、输入和输出
# 需要先在HuggingFace个人账户侧上传自定义的数据集，然后采用下述方法，从HuggingFace加载到colab中
EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # 必须添加EOS_TOKEN，否则无限生成
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts, }

from datasets import load_dataset
dataset = load_dataset("Jasonniu/ruozhiba_LLaMA3", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True, )


# Step3、配置 LoRA、训练并导出模型
# 配置LoRA参数和训练参数
from trl import SFTTrainer
from transformers import TrainingArguments

# 获取并配置LoRA优化的语言模型
model = FastLanguageModel.get_peft_model(
    model,  # 原始模型
    r=16,   # LoRA优化的秩，建议值为8, 16, 32, 64, 128
    # LoRA将应用于这些模块
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",        # 不使用偏置
    use_gradient_checkpointing="unsloth",  # 使用梯度检查点，适用于长文本
    random_state=3407,
    use_rslora=False,   # 不使用秩稳定 LoRA（Rank-Stabilized LoRA）
    loftq_config=None,  # LoFT-Q配置，此处为None
)

# 初始化SFTTrainer训练器
FastLanguageModel.for_training(model)
trainer = SFTTrainer(
    model=model,                 # 已配置的模型
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",   # 数据集中文本字段名称
    max_seq_length=max_seq_length,
    dataset_num_proc=2,          # 数据预处理进程数
    packing=False,               # 不将多个短样本打包到同一训练序列。
    args=TrainingArguments(
        per_device_train_batch_size=2,  # 每个设备上的训练批次大小
        gradient_accumulation_steps=4,  # 梯度累积步数
        warmup_steps=5,                 # 预热步数
        max_steps=60,                   # 微调步数
        learning_rate=2e-4,             # 学习率
        fp16=not torch.cuda.is_bf16_supported(), # 使用FP16精度，如果BF16不支持
        bf16=torch.cuda.is_bf16_supported(),     # 使用BF16精度，如果支持
        logging_steps=1,                # 日志记录步数
        optim="adamw_8bit",             # 使用8位AdamW优化器
        weight_decay=0.01,              # 权重衰减
        lr_scheduler_type="linear",     # 学习率调度器类型
        seed=3407,
        output_dir="outputs",           # 合并模型的输出目录
    ),
)


# 开始训练
trainer_stats = trainer.train()

# 测试微调后的模型
# 基于中文提示进行文本回答，并通过TextStreamer实时输出生成的文本
FastLanguageModel.for_inference(model)
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "只用中文回答问题",  # instruction：用户指令
            "火烧赤壁 曹操为何不拨打119求救？",  # input：补充输入
            "",  # output：模型回答
        )
    ], return_tensors="pt").to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)


# 保存LoRA模型
model.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # 在线保存到hugging face，需要token


# 合并LoRA模型并量化和转换格式：为hf格式，然后4位量化并转换gguf格式
# 合并原始模型和LoRA模型(16位的hf格式)【16GB】，其次转换为16位的gguf格式【16GB】，最后量化成4位gguf保存【4GB】
# 因此合并量化模型，至少要有40G的空间，16+16+4
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
# model.save_pretrained_merged("outputs", tokenizer, save_method = "merged_16bit",) #合并模型，保存为16位hf
# model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "") #合并4位gguf，上传到hugging face(需要账号token)


# Step4、复制导出的模型文件
# 注意，由于直接右键下载很慢，可先挂载google云盘，然后通过google网盘下载就很快
# 先挂载google云盘
from google.colab import drive
from pathlib import Path
import shutil

code_root = Path.cwd()
drive.mount(str(code_root / 'drive'))

# 复制模型到google云盘
# save_pretrained_gguf 将文件写入指定的 model 目录。
source_files = sorted((code_root / "model").glob("*.gguf"))
if not source_files:
    raise FileNotFoundError("model 目录中没有导出的 GGUF 文件")
destination_dir = code_root / 'drive/MyDrive/Llama3'
destination_dir.mkdir(parents=True, exist_ok=True)
for source_file in source_files:
    shutil.copy(source_file, destination_dir / source_file.name)
