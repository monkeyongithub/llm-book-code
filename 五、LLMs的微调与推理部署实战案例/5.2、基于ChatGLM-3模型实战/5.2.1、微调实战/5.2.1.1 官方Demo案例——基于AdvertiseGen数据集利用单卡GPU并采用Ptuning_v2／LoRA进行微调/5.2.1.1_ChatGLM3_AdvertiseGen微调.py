# 文件用途：演示AdvertiseGen微调
# 对应章节/小节：第5章 5.2.1.1 官方Demo案例——基于AdvertiseGen数据集利用单卡GPU并采用Ptuning_v2/LoRA进行微调
# 作者：NYY
# 创建日期：2024-10-11（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# 环境准备
# # 0.1、环境配置
# >> 本机硬件
# 显存：推荐至少24GB 显卡、 16GB内存
#
# >> 下载项目仓库：
# git clone https://github.com/THUDM/ChatGLM3
# cd ChatGLM3
#
#
# >> 安装依赖
# 注意：推荐安装Anaconda，以管理Python环境和依赖库
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# torch 推荐使用 2.0 及以上的版本，以获得最佳的推理性能。
# transformers>=4.30.2
# jieba>=0.42.1
# ruamel_yaml>=0.18.6
# rouge_chinese>=1.0.3
# jupyter>=1.0.0
# datasets>=2.18.0
# peft>=0.10.0
# deepspeed==0.13.1
# mpi4py>=3.1.5
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
# >> 下载模型
# 模型地址：https://huggingface.co/THUDM/chatglm3-6b



# 完整实现
# Step1、将 AdvertiseGen 转换为对话样本
# 官方要求的数据集格式【conversations-user-assistant】：对话能力而非工具能力
# {"conversations":
#      [{"role": "user", "content": "类型#裙*裙长#半身裙"},
#       {"role": "assistant", "content": "这款百搭时尚的仙女半身裙，整体设计非常的飘逸随性，穿上之后每个女孩子都能瞬间变成小仙女啦。料子非常的轻盈，透气性也很好，穿到夏天也很舒适。"}
#       ]
#  }
# 将下载的数据集转为上述格式并保存到本地
# 下述代码定义了一个用于转换数据的函数 convert_adgen，它读取指定目录下的 JSON 文件（train.json 和 dev.json），并将每行数据转换为新的格式，包含用户和助手的对话内容。内部使用了 _resolve_path 函数处理路径，确保路径正确，而 _mkdir 函数则确保输出目录存在。最终，转换后的数据被写入指定的保存目录。
import json
from typing import Union
from pathlib import Path

# 利用辅助函数 _resolve_path，将输入的路径参数转换为 Path 对象，扩展用户目录并解析符号链接。
def _resolve_path(path: Union[str, Path]) -> Path:  # 采用类型注解法(对参数和返回值注释)→提高代码的可维护性和可读性
    return Path(path).expanduser().resolve()

# 利用辅助函数 _mkdir创建目录
def _mkdir(dir_name: Union[str, Path]):
    dir_name = _resolve_path(dir_name)
    if not dir_name.is_dir():
        dir_name.mkdir(parents=True, exist_ok=False)

# 读取给定目录中的JSON文件，对每一行的JSON数据进行解析和转换，然后将结果保存到指定目录。
def convert_adgen(data_dir: Union[str, Path], save_dir: Union[str, Path]):
    def _convert(in_file: Path, out_file: Path):
        _mkdir(out_file.parent)
        with open(in_file, encoding='utf-8') as fin:
            with open(out_file, 'wt', encoding='utf-8') as fout:
                for line in fin:
                    dct = json.loads(line)
                    sample = {'conversations': [{'role': 'user', 'content': dct['content']},
                                                {'role': 'assistant', 'content': dct['summary']}]}
                    fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

    data_dir = _resolve_path(data_dir)
    save_dir = _resolve_path(save_dir)

    train_file = data_dir / 'train.json'
    if train_file.is_file():
        out_file = save_dir / train_file.relative_to(data_dir)
        _convert(train_file, out_file)

    dev_file = data_dir / 'dev.json'
    if dev_file.is_file():
        out_file = save_dir / dev_file.relative_to(data_dir)
        _convert(dev_file, out_file)
convert_adgen('data/AdvertiseGen', 'data/AdvertiseGen_fix')

# 转换后的结果如下所示：多轮conversations(user-assistant)
# {
#     "conversations": [
#         {
#             "role": "user",
#             "content": "用户的问题或请求"
#         },
#         {
#             "role": "assistant",
#             "content": "助手的回答或回复"
#         }
#     ]
# }



# Step2、配置并执行 ChatGLM-3 微调
# 采用命令行执行微调：支持SFT/Ptuning_v2/LoRA等微调，此处以LoRA微调为例
# !CUDA_VISIBLE_DEVICES=1 venv/bin/python3 finetune_hf.py  data/AdvertiseGen_fix Data/Models/LLM/chatglm3-6b  configs/lora.yaml

# LLMs之ChatGLM-3：ChatGLM3源码解读(finetune_hf.py)微调模型(评估/保存)实现生成任务——配置初始化(加载配置信息、分词器和模型(配置PEFT参数)、初始化数据管理器)→数据预处理(分词+截断+批处理+检查一致性)→模型准备和配置(采用梯度检查点技术)→模型微调(支持SFT/Ptuning_v2/LoRA+可从检查点恢复)→模型评估(BLEU/ROUGE等)→模型测试
# https://yunyaniu.blog.csdn.net/article/details/136441045


# 对比三大微调方式的区别：sft.yaml、ptuning_v2.yaml、lora.yaml参数对比
# ptuning_v2.yaml
# peft_config:
#   peft_type: PREFIX_TUNING
#   task_type: CAUSAL_LM
#   num_virtual_tokens: 128

# lora.yaml
# peft_config:
#   peft_type: LORA
#   task_type: CAUSAL_LM
#   r: 8
#   lora_alpha: 32
#   lora_dropout: 0.1



# Step3、加载检查点测试生成
# 在完成微调任务之后，我们可以查看到 output 文件夹下多了很多个checkpoint-*的文件夹，这些文件夹代表了训练的轮数。 我们选择最后一轮的微调权重，并使用inference进行导入。
# !ls output/
# !CUDA_VISIBLE_DEVICES=1  python inference_hf.py output/checkpoint-3000/ --prompt "类型#裙*版型#显瘦*材质#网纱*风格#性感*裙型#百褶*裙下摆#压褶*裙长#连衣裙*裙衣门襟#拉链*裙衣门襟#套头*裙款式#拼接*裙款式#拉链*裙款式#木耳边*裙款式#抽褶*裙款式#不规则"
