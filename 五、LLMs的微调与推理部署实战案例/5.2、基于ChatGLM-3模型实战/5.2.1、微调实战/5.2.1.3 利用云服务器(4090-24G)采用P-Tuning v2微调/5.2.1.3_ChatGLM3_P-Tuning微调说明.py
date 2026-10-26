# 文件用途：说明P-Tuning v2微调
# 对应章节/小节：第5章 5.2.1.3 利用云服务器(4090-24G)采用P-Tuning v2微调
# 作者：NYY
# 创建日期：2024-10-11（文件系统记录）
# 运行方式：阅读示例说明
# 备注：操作说明，不包含独立执行逻辑

# 环境准备
# # 0.1、环境准备
# >> 选择一家价格合适的云服务商
# 可以采用按时、按时、按月等收费方式
# Python=3.10，PyTorch=2.0.0，cuda=12.2
#
# >> 本机硬件
# GPU RTX 4090-24GB，CPU 12核，内存80GB；
# 系统盘30GB、数据盘50GB；
#
# >> 下载项目仓库：
# git clone https://github.com/THUDM/ChatGLM3
# cd ChatGLM3
# 注意，此处也可以直接复制镜像而不必逐个下载
#
#
# >> 创建并激活虚拟环境
# 注意：推荐安装Anaconda，以管理Python环境和依赖库
# conda create --name LLMs_chat_DIY python=3.10
# cd ChatGLM3_6b/fine_tune
# source activate LLMs_chat_DIY
#
# >> 安装依赖
# sudo pip install -r requirements.txt
#
#
# # 0.2、文件下载
# >> 下载数据集
# (1)、AdvertiseGen数据集：可以从 Google Drive 或者 Tsinghua Cloud 下载处理好的 AdvertiseGen 数据集。
# 数据集地址(源自Tsinghua)：https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1
# 数据集地址(源自Google)：https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing
# 原始数据集格式：【content-summary】地址：https://huggingface.co/datasets/shibing624/AdvertiseGen/viewer
# 示例如下所示：
# {
#   "content": "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞",
#   "summary": "简约而不简单的牛仔外套，白色的衣身十分百搭。衣身多处有做旧破洞设计，打破单调乏味，增加一丝造型看点。衣身后背处有趣味刺绣装饰，丰富层次感，彰显别样时尚。"
# }
#
# (2)、自测数据集地址：https://github.com/hiyouga/LLaMA-Factory/blob/main/data/self_cognition.json
# 或者https://github.com/hiyouga/LLaMA-Factory/blob/main/data/identity.json
#
# >> 下载模型
# 模型地址：https://huggingface.co/THUDM/chatglm3-6b
#
#
# # 0.3、测试基于Gradio的网页对话界面
# python web_demo.py
# 注意，如果可以打开界面，则说明当前环境配置成功！





# 完整实现
# Step1、准备指令数据
# 根据要微调想要的能力选择不同格式的数据集。比如利用python脚本实现转换，其中format_advertise_gen.py文件将数据集转换为单论对话格式，format_tool_alpaca.py文件将数据集转换为工具调用格式。

# 数据预处理：将三元组格式【instruction-input-output】转为二元组格式【prompt-response】
# ./scripts/format_advertise_gen.py --path "AdvertiseGen/train.json"


# Step2、配置并运行 P-Tuning v2 微调
# source activate LLMs_chat_DIY
# ./scripts/finetune_pt.sh  # P-Tuning v2 微调
# 微调命令调用项目中的 finetune.py。


# Step3、加载检查点测试对话
# source activate LLMs_chat_DIY
# python inference.py --pt-checkpoint "output/advertise_gen_pt-20231120-191528-128-2e-2" --model models/chatglm3-6b
