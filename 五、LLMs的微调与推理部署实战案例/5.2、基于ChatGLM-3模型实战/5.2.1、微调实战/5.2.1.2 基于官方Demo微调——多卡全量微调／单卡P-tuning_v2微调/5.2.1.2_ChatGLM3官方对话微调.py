# 文件用途：说明官方对话微调
# 对应章节/小节：第5章 5.2.1.2 基于官方Demo微调——多卡全量微调/单卡P-tuning_v2微调
# 作者：NYY
# 创建日期：2024-10-11（文件系统记录）
# 运行方式：阅读示例说明
# 备注：操作说明，不包含独立执行逻辑

'''
基于官方Demo微调：单论对话对微调+多轮对话微调【可包括工具调用能力】
源码地址：https://yunyaniu.blog.csdn.net/article/details/137412422

'''


# 环境准备
# 环境与模型准备参见书稿对应小节。



# 完整实现
# Step1、准备对话数据集
# 单论对话对微调数据集：数据格式为【prompt-response】
# [
#   {
#     "prompt": "<提示文本>",
#     "response": "<回答文本>"
#   }
#   // ...
# ]


# B、多轮对话微调数据集：数据格式为【conversations-system-user】
# [ {
#     "conversations": [
#       {
#         "role": "system", "content": "<系统提示文本>"
#       },
#       {
#         "role": "user", "content": "<用户输入文本>"
#       },
#       {
#         "role": "assistant", "content": "<助手回答文本>"
#       },
#       // ……继续添加后续轮次的对话
#       {
#         "role": "user", "content": "<用户输入文本>"
#       },
#       {
#         "role": "assistant", "content": "<助手回答文本>"
#       } ] }
#   // ...
# ]


# C、多轮对话微调【带工具调用能力】数据集：数据格式为【tools-conversations-system-user-assistant】
# [ {
#     "tools": [
#       // 可用工具的描述，此处不限定具体格式
#     ],
#     "conversations": [
#       {
#         "role": "system", "content": "<系统提示文本>"
#       },
#       {
#         "role": "user", "content": "<用户输入文本>"
#       },
#       {
#         "role": "assistant", "content": "<助手调用工具前的说明>"
#       },
#       {
#         "role": "tool",
#         "name": "<待调用工具的名称>",
#         "parameters": {
#           "<参数名称>": "<参数值>"
#         },
#         "observation": "<工具返回结果>"
#         // 工具返回结果不必是字符串
#       },
#       {
#         "role": "assistant", "content": "<助手根据工具结果生成的回答>"
#       },
#       // ……继续添加后续轮次的对话
#       {
#         "role": "user", "content": "<用户输入文本>"
#       },
#       {
#         "role": "assistant", "content": "<助手回答文本>"
#       } ]  }
#   // ...
# ]


# Step2、配置并执行 ChatGLM-3 微调
# 采用命令行执行微调：支持多卡全量微调ds/单卡Ptuning_v2等微调，此处以Ptuning_v2微调为例
# >> 查看脚本内容
# ./scripts/finetune_pt.sh

#! /usr/bin/env bash
# set -ex
# PRE_SEQ_LEN=128
# LR=2e-2
# NUM_GPUS=1
# MAX_SOURCE_LEN=1024
# MAX_TARGET_LEN=128
# DEV_BATCH_SIZE=1
# GRAD_ACCUMULARION_STEPS=32
# MAX_STEP=100
# SAVE_INTERVAL=50

# DATESTR=`date +%Y%m%d-%H%M%S`
# RUN_NAME=advertise_gen_pt

# BASE_MODEL_PATH=models/chatglm3-6b
# DATASET_PATH=data_DIY/self_cognition_prompt_response_DIY.json
# OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${PRE_SEQ_LEN}-${LR}

# mkdir -p $OUTPUT_DIR
# torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
#     --train_format input-output \
#     --train_file $DATASET_PATH \
#     --preprocessing_num_workers 1 \
#     --model_name_or_path $BASE_MODEL_PATH \
#     --output_dir $OUTPUT_DIR \
#     --max_source_length $MAX_SOURCE_LEN \
#     --max_target_length $MAX_TARGET_LEN \
#     --per_device_train_batch_size $DEV_BATCH_SIZE \
#     --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
#     --max_steps $MAX_STEP \
#     --logging_steps 1 \
#     --save_steps $SAVE_INTERVAL \
#     --learning_rate $LR \
#     --pre_seq_len $PRE_SEQ_LEN 2>&1 | tee ${OUTPUT_DIR}/train.log


# >> 解读finetune.py文件
# LLMs之ChatGLM-3：finetune.py源码解读微调模型—参数解析与初始化配置(清空CUDA缓存+解析JSON文件中的参数+设置训练日志格式和级别+设置随机种子)→加载预训练模型和分词器(支持P-tuning v2技术和精度量化技术)→数据预处理(加载json或jsonl格式书籍+支持不同训练格式数据集【输入-输出格式/多轮对话格式】+检查训练数据集中第一个样本的基本正确性(只在主进程))→初始化数据收集器和训练器→模型训练(支持继续训练+梯度检查点技术+执行训练，显存21G/对比推理需要13G)→保存模型及状态
# https://yunyaniu.blog.csdn.net/article/details/134985534



# 对比两大微调方式的区别：finetune_ds.sh、finetune_pt.sh参数对比
# 两个脚本均有调用 torchrun 命令来启动分布式过程
# ./scripts/finetune_ds.sh
# 适用于更大规模的训练任务，使用了更多的GPU、更低的学习率、更少的梯度累积步数和更长的训练时间，并且应用了混合精度和 deepspeed 技术以加速训练。

# ./scripts/finetune_pt.sh
# 配置了更小的GPU资源、更高的学习率、更多的梯度累积步数和更短的训练时间，且加入了新的参数 PRE_SEQ_LEN。这种配置可能更适合较小的模型或资源受限的环境。
