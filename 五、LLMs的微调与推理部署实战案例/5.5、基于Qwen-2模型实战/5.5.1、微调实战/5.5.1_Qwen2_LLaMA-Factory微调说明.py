# 文件用途：说明Qwen-2的LLaMA-Factory微调
# 对应章节/小节：第5章 5.5.1 微调实战——基于LLaMA-Factory框架微调Qwen-2模型
# 作者：NYY
# 创建日期：2024-10-14（文件系统记录）
# 运行方式：阅读示例说明
# 备注：操作说明，不包含独立执行逻辑

# 环境准备
# # 0.1、环境配置
# >> 本机硬件
# 显存：
#
# >> 安装LLaMA-Factory：
# git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
# cd LLaMA-Factory
#
#
# >> 安装依赖
# pip install deepspeed
# pip install flash-attn --no-build-isolation
# 注意，如你使用 FlashAttention-2 ，请确保你的CUDA版本在11.6以上。
#
#
# # 0.2、文件下载
# >> 下载本仓库：
#
# >> 下载数据集
#
#
# >> 下载模型



# 完整代码
# 源码地址：https://qwen.readthedocs.io/zh-cn/latest/training/SFT/llama_factory.html

# Step1、准备 Qwen-2 微调数据
# 比如alpaca格式的数据集应遵循以下格式：
# [
#   {
#     "instruction": "用户指令（必填）",
#     "input": "补充输入（可选）",
#     "output": "模型回答（必填）",
#     "system": "系统提示（可选）",
#     "history": [
#       ["第一轮用户指令（可选）", "第一轮模型回答（可选）"],
#       ["第二轮用户指令（可选）", "第二轮模型回答（可选）"]
#     ]
#   }
# ]


# Step2、配置并执行 LLaMA-Factory 训练
# DISTRIBUTED_ARGS="
#     --nproc_per_node $NPROC_PER_NODE    # 每个节点上使用的进程数（GPU数）
#     --nnodes $NNODES \                    # 参与分布式训练的节点总数
#     --node_rank $NODE_RANK \              # 当前节点的排名或ID
#     --master_addr $MASTER_ADDR \          # 主节点的IP地址，用于初始化分布式环境
#     --master_port $MASTER_PORT            # 主节点上的端口号，用于通信。
#   "
# torchrun $DISTRIBUTED_ARGS src/train.py \  # torchrun是在多个GPU或节点上运行Python脚本的命令行工具
#     --deepspeed $DS_CONFIG_PATH \          # DeepSpeed配置文件的路径
#     --stage sft \                          # 指定训练阶段为“sft”
#     --do_train \                           #  执行训练任务
#     --use_fast_tokenizer \                 # 使用快速分词器
#     --flash_attn \                         # 使用FlashAttention机制，这是一种加速注意力计算的方法。
#     --model_name_or_path $MODEL_PATH \     # 
#     --dataset your_dataset \
#     --template qwen \                      # 训练数据集的模板或策略，这里指定为 “qwen”。
#     --finetuning_type lora \               # 微调类型设置为LoRA
#     --lora_target q_proj,v_proj\           # 指定lora微调的目标层，这里是 “q_proj” 和 “v_proj”。
#     --output_dir $OUTPUT_PATH \
#     --overwrite_cache \          
#     --overwrite_output_dir \               # 如果为True，则覆盖输出目录中的现有内容。
#     --warmup_steps 100 \                   # 学习率预热步骤数，即在开始时逐渐增加学习率。
#     --weight_decay 0.1 \                   # 权重衰减系数，用于L2正则化
#     --per_device_train_batch_size 4 \      # 每个GPU的训练批次大小
#     --gradient_accumulation_steps 4 \      # 梯度累积步数，可以有效地增大批量大小而不消耗更多内存。
#     --ddp_timeout 9000 \                   # 分布式数据并行（DDP）的初始化超时时间
#     --learning_rate 5e-6 \    
#     --lr_scheduler_type cosine \           # 学习率调度器类型为余弦退火
#     --logging_steps 1 \                    # 日志记录间隔，每几步打印一次训练信息
#     --cutoff_len 4096 \                    # 截断长度，超过此长度的数据将被截断。
#     --save_steps 1000 \                    # 保存模型检查点的步数间隔
#     --plot_loss \                          # 是否绘制损失图
#     --num_train_epochs 3 \                 # 训练轮数
#     --bf16                                 # 使用bfloat16精度进行训练，可以加速训练同时减少内存使用
# 通过修改训练命令中的参数来调整超参数。其中一个需要注意的参数是 cutoff_len ，它代表训练数据的最大长度。通过控制这个参数，可以避免出现OOM（内存溢出）错误。


# Step3、合并 LoRA 参数并导出
# 如果你使用 LoRA 训练模型，可能需要将adapter参数合并到主分支中。请运行以下命令以执行 LoRA adapter 的合并操作。
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
#     --model_name_or_path path_to_base_model \
#     --adapter_name_or_path path_to_adapter \
#     --template qwen \
#     --finetuning_type lora \
#     --export_dir path_to_export \
#     --export_size 2 \
#     --export_legacy_format False
