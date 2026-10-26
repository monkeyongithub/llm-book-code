# 文件用途：说明LLaMA-Factory微调
# 对应章节/小节：第5章 5.2.1.4 基于LLaMA-Factory框架并采用GUI方式微调
# 作者：NYY
# 创建日期：2024-10-13（文件系统记录）
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
# 注意：推荐安装Anaconda，以管理Python环境和依赖库
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#
# # 0.2、文件下载
# >> 下载数据集
# 自测数据集地址：https://github.com/hiyouga/LLaMA-Factory/blob/main/data/self_cognition.json
# 或者https://github.com/hiyouga/LLaMA-Factory/blob/main/data/identity.json
#
#
# >> 下载模型
# 模型地址：https://huggingface.co/THUDM/chatglm3-6b



# 完整代码
# Step1、启动 LLaMA-Factory 图形界面
# 启动LLaMA-Factory的web页面
# python src/train_web.py


# Step2、设置参数并开始微调
# 在GUI界面配置微调参数并启动训练，此外，可以点击“预览命令”，可以看到要执行的python脚本，示例如下
# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --model_name_or_path LLaMA-Factory-main/chatglm3\
#     --dataset alpaca_gpt4_zh \


# Step3、选择适配器测试推理
# 在GUI界面选择基于微调好的模型进行推理测试
