# 文件用途：说明GLM-4微调流程
# 对应章节/小节：第5章 5.4.1.2 利用云服务器(4090-24G)采用LoRA或P-Tuning v2实现对话微调
# 作者：NYY
# 创建日期：2024-10-13（文件系统记录）
# 运行方式：阅读示例说明
# 备注：操作说明，不包含独立执行逻辑

# 环境准备
# # 0.1、环境准备
# >> 选择一家价格合适的云服务商
# 可以采用按时、按时、按月等收费方式
# Python=3.10，PyTorch=2.0.0，cuda=12.0
#
# >> 本机硬件
# GPU RTX 4090-24GB，CPU 15核，内存80GB；
# 系统盘30GB、数据盘50GB；
#
# >> 下载项目仓库：
# git clone https://github.com/THUDM/GLM-4
# cd GLM-4
# 注意，此处也可以直接复制镜像而不必逐个下载
#
#
# >> 创建并激活虚拟环境
# 注意：推荐安装Anaconda，以管理Python环境和依赖库
# conda create --name py310_chat python=3.10
# cd GLM-4/finetune_demo
# source activate py310_chat
# 源码地址：https://github.com/THUDM/GLM-4/tree/main/finetune_demo
#
#
# >> 安装依赖
# pip install -r requirements.txt
#
#
# # 0.2、文件下载
# >> 下载数据集
# (1)、自测数据集参考地址：https://github.com/hiyouga/LLaMA-Factory/blob/main/data/self_cognition.json
# 或者https://github.com/hiyouga/LLaMA-Factory/blob/main/data/identity.json
#
#
# >> 下载模型
# 模型地址：https://huggingface.co/THUDM/glm-4-9b



# 完整代码
# Step1、转换问答数据为 ChatML
# chcp 65001
# 数据转换脚本 5.4.1.2_问答数据转ChatML.py 与本文件同目录。
# 通过 --input_file 指定微调实战目录 data 下的实际问答文件。


# Step2、配置云端微调并测试生成
# T1、采用LoRA实现微调：
# 模型微调
# bash DIY/single_turn_dialogue/LoRA/lora_train.sh
# source activate py310_chat     
# cd GLM-4/finetune_demo
# python finetune.py  data/  models/glm-4-9b-chat  configs/lora.yaml


# 模型推理：测试对话功能
# bash DIY/single_turn_dialogue/LoRA/lora_inference.sh
# source activate py310_chat     
# cd GLM-4/finetune_demo/inference
# python lora_inference.py






# T2、采用P-Tuning v2实现微调：训练阶段(耗费显存更高)，推理阶段(易过拟合、易连续不断吐出文本)

# 模型微调
# bash DIY/ptuning_v2/ptuning_v2_train.sh
# source activate py310_chat     
# cd GLM-4/finetune_demo
# python finetune.py  data/  models/glm-4-9b-chat  configs/ptuning_v2.yaml
# 注意，ptuning_v2微调比LoRA耗费显存更高。ptuning_v2训练很容易出现过拟合的情况，如果训练得当，ptuning_v2可以取得非常优异的效果，即表现的上限和下限差距较大。
# 模型有时可能会出现连续不断吐出文本的现象，针对这种情况，最好直接终止训练。



# 模型推理
# bash DIY/single_turn_dialogue/ptuning_v2/ptuning_v2_inference.sh
# source activate py310_chat     
# cd GLM-4/finetune_demo/inference
# python ptuning_v2_inference.py
