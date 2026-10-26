# 文件用途：说明GLM-4微调流程
# 对应章节/小节：第5章 5.4.1.1 基于官方demo案例依次采用lora(单机单卡)、p-tuning v2(单机单卡)、SFT(单机8卡)实现微调
# 作者：NYY
# 创建日期：2024-10-20（文件系统记录）
# 运行方式：阅读示例说明
# 备注：操作说明，不包含独立执行逻辑

# 本节示例：基于官方demo案例依次采用lora(单机单卡)、p-tuning v2(单机单卡)、SFT(单机8卡)实现微调
'''
源码地址：https://github.com/THUDM/GLM-4/tree/main/finetune_demo
'''

# 环境准备
# 环境配置
# >> 本机硬件
# 显存：NVIDIA A100-80GB*8，内存 512GB，GPU 驱动版本 535.104.05
# Ubuntu 22.04或Debian，Python 3.10，CUDA 12.3
# 以及支持 BF16推理的 GPU 设备

# >> 显存占用说明
# 微调模型	微调方案	显存占用	权重保存点大小
# GLM-4-9B-Chat	lora (PEFT)	22G	17M
# GLM-4-9B-Chat	p-tuning v2 (PEFT)	21G	121M
# GLM-4-9B-Chat	SFT（ZeRO-3 方案）	80G（每张 GPU，需要使用 8 张 GPU）	20G


# >> 安装依赖
# pip install -r requirements.txt
# jieba==0.42.1
# datasets==2.20.0
# peft==0.12.2
# deepspeed==0.14.4
# nltk==3.8.1
# rouge_chinese==1.0.3
# ruamel.yaml==0.18.6



# 文件下载
# >> 下载本仓库：
# git clone https://github.com/THUDM/GLM-4
# cd GLM-4





# >> 下载数据集
# AdvertiseGen数据集地址：


# >> 下载模型
# 模型地址：https://huggingface.co/THUDM/glm-4-9b




# 完整代码

# Step1、准备微调数据集
# 数据集格式示例——仅微调模型的对话能力，而非工具能力：【messages-system-user-assistant】
# {
#   "messages": [
#     {
#       "role": "user", "content": "类型#裤*材质#牛仔布*风格#性感"
#     },
#     {
#       "role": "assistant", "content": "3x1的这款牛仔裤采用浅白的牛仔面料为裤身材质，其柔然的手感和细腻的质地，在穿着舒适的同时，透露着清纯甜美的个性气质。除此之外，流畅的裤身剪裁将性感的腿部曲线彰显的淋漓尽致，不失为一款随性出街的必备单品。"
#     }
#   ]
# }

# 数据集格式示例——微调模型的对话能力+工具调用能力：【messages-system-tools-user-assistant-observation-assistant】
# 注意事项
# system 角色为可选角色，但若存在 system 角色，其必须出现在 user 角色之前，且一个完整的对话数据（无论单轮或者多轮对话）只能出现一次 system 角色。
# tools 字段为可选字段，若存在 tools 字段，其必须出现在 system 角色之后，且一个完整的对话数据（无论单轮或者多轮对话）只能出现一次 tools 字段。
# 当 tools 字段存在时，system 角色必须存在并且 content 字段为空。
# {
#   "messages": [
#     {
#       "role": "system", "content": "",
#       "tools": [
#         {
#           "type": "function",
#           "function": {
#             "name": "get_recommended_books",
#             "description": "根据用户的兴趣推荐书籍",
#             "parameters": {
#               "type": "object",
#               "properties": {
#                 "interests": {
#                   "type": "array",
#                   "items": {
#                     "type": "string"
#                   },
#                   "description": "根据这些兴趣推荐书籍"
#                 }
#               },
#               "required": [
#                 "interests"
#               ]
#             }
#           }
#         }
#       ]
#     },
#     {
#       "role": "user", "content": "你好，我想要一些书籍推荐。我对历史和科幻感兴趣。"
#     },
#     {
#       "role": "assistant", "content": "{\"name\": \"get_recommended_books\", \"arguments\": {\"interests\": [\"历史\", \"科幻\"]}}"
#     },
#     {
#       "role": "observation", "content": "{\"books\": [\"尤瓦尔·赫拉利的《人类简史》\", \"史蒂芬·霍金的《时间简史》\", \"弗兰克·赫伯特的《沙丘》\", \"安迪·威尔的《火星救援》\"]}"
#     },
#     {
#       "role": "assistant", "content": "根据你对历史和科幻的兴趣，我推荐以下书籍：《人类简史》——尤瓦尔·赫拉利，《时间简史》——史蒂芬·霍金，《沙丘》——弗兰克·赫伯特，以及《火星救援》——安迪·威尔。"
#     }
#   ]
# }


# Step2、按所选方案配置并运行 GLM-4 微调
# 配置文件
# (1)、deepspeed 配置文件
# ds_zereo_2 / ds_zereo_3.json
# 在Stage 2中，优化器状态是分割的，并且分布在多个GPU上，而梯度仍然通过all-reduce操作来聚合。
# 在Stage 3中，设为AdamW，除了Stage 2的特性外，参数也进行了分割(如sub_group_size)，并且每个GPU只保存部分参数，从而进一步减少每个GPU的内存需求。
# ds_zero_2.json到ds_zero_3.json的转变代表了从优化器状态分片到模型参数和优化器状态都进行分片的过程，以及提供了更多高级选项来优化内存效率和计算性能(新增stage3_为前缀的一系列参数)。

# (2)、peft配置文件
# lora.yaml / ptuning_v2.yaml / sft.yaml: 模型不同方式的配置文件，包括模型参数、优化器参数、训练参数等。 部分重要参数解释如下：
# peft_config 部分
#     peft_type: 使用的参数有效调整类型 (支持 LORA 和 PREFIX_TUNING)。
#     task_type: 任务类型，这里是因果语言模型 (不要改动)。
# Lora参数：
#     r: LoRA 的秩。
#     lora_alpha: LoRA 的缩放因子。
#     lora_dropout: 在 LoRA 层使用的 dropout 概率。
# P-TuningV2参数：
#     num_virtual_tokens: 虚拟 token 的数量。
#     num_attention_heads: 设置为2，P-TuningV2 的注意力头数(不要改动)。
#     token_dim: 设置为256，P-TuningV2 的 token 维度(不要改动)。

# 模型微调：采用lora方式微调Chat模式
# 注意：如果采用其它微调技术，替换lora.yaml文件即可。比如将lora.yaml替换为ptuning_v2.yaml或者sft.yaml

# T1、单机单卡
# 通过以下代码执行 单机单卡 运行
# python finetune.py  data/AdvertiseGen/  THUDM/glm-4-9b-chat configs/lora.yaml

# 如果你想从训练一半的模型开始微调，而不是每次微调都从头开始，可以加入第四个参数
# python finetune.py  data/AdvertiseGen/  THUDM/glm-4-9b-chat  configs/lora.yaml yes   # yes表示自动从最后一个保存的 Checkpoint开始训练

# GLM-4-9B源码解读(finetune.py)模型微调/评估/预测的完整实现——解析命令行参数→加载(微调配置器【确保加载参数和逻辑正确性】/模型【必须BF16】&分词器并支持PEFT配置/数据管理器【指定目录加载和按需处理】)→定义并处理数据集(训练集【基于每个对话生成训练数据(输入序列和标签)并处理损失掩码】/验证集&测试集【基于每个对话生成评估或推理所需的输入-输出对】)→模型训练(设置训练属性【梯度检查点/输入梯度计算—为匹配p-tuning v2技术】→创建训练器(Seq2SeqTrainer实例【返回loss&生成token&标签】/数据末尾填充和匹配长度/配置训练参数/数据整理器/计算指标【ROUGE和BLEU分数—rouge-1/rouge-2/rouge-l/bleu-4】)→训练模型【支持从检查点恢复训练】)→模型评估(若存在则预测)


# T2、单机8卡：通过 torchrun 在单机 8 卡上使用 deepspeed 来加速模型的微调过程
# 通过以下代码执行 单机多卡/多机多卡 运行，这是使用 deepspeed 作为加速方案的，您需要安装 deepspeed。接着，按照此命令运行：
# OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=8  finetune.py  data/AdvertiseGen/  THUDM/glm-4-9b-chat  
#                           configs/lora.yaml # 用于对话微调

# 解读参数
# OMP_NUM_THREADS=1：设置 OpenMP并行编程 的线程数为 1，避免在多卡运行中，使用太多 CPU 线程，从而影响到 GPU 性能
# torchrun：支持多机多卡训练
# --standalone：这个参数表示运行的分布式环境是“独立”模式，即所有节点都是平等的，不需要额外的 master 节点协调进程。通常用于单机多卡的情况。
# --nnodes=1：nnodes 表示参与训练的计算节点数。这里 nnodes=1 表示只有一台机器参与训练。如果是多机训练，可以把 nnodes 设为机器数。
# --nproc_per_node=8：这个参数表示每个节点(该机器)使8张GPU，并且会在每张 GPU 上启动一个训练进程。对于多卡训练，nproc_per_node 通常等于机器上的 GPU 数量。


# 模型推理
# python inference.py your_finetune_path
# 注意：请注意，该案例对于 LORA 和 P-TuningV2没有合并训练后的模型，而是在adapter_config.json 中记录了微调型的路径；
# 如果你的原始模型位置发生更改，则你应该修改adapter_config.json中base_model_name_or_path的路径。
