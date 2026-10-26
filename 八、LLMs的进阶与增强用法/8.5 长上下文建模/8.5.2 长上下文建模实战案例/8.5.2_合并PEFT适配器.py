# 文件用途：演示多个PEFT适配器合并
# 对应章节/小节：第8章 8.5.2 长上下文建模实战案例
# 作者：NYY
# 创建日期：2024-11-01（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

'''
LLMs之LLaMA-3：源代码解读merge_adapters.py(仅需58行代码)合并多个PEFT模型(LoRA技术)将LLaMA-3扩展到100万/1048k上下文——解析命令行参数→在基础模型上循环加载LoRA模型→加载分词器模型→将合并后的模型以及分词器保存到指定目录或推送到模型中心
源代码地址：https://gist.github.com/ehartford/731e3f7079db234fa1b79a01e09859ac
这段代码提供了一个工具，允许用户将多个PEFT模型合并到一个基础模型上，从而实现模型能力的扩展。
PEFT是一种参数高效的微调方法，它允许在不修改基础模型参数的情况下，通过添加少量的参数来适应新的任务。
这个脚本简化了合并过程，并提供了将合并后的模型保存到本地或推送到模型中心的功能。通过这种方式，用户可以轻松地在不同的微调任务之间迁移和组合模型改进。
'''

# 支持依次合并多个与基础模型兼容的 PEFT 适配器。
# python merge_adapters.py --base_model_name_or_path <base_model> --peft_model_paths <adapter1> <adapter2> <adapter3> --output_dir <merged_model>

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
import argparse

# 合并只叠加兼容的适配器权重；长上下文能力还取决于基础配置与适配器的训练。
# 不能仅凭 merge_and_unload 宣称支持百万 token。

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, required=True)
    parser.add_argument("--peft_model_paths", type=str, required=True, nargs='+', help="List of paths to PEFT models")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()

def main():
    # Step1、解析基础模型、适配器和输出参数
    args = get_args()

    # Step2、加载基础模型并逐个合并适配器
    # 加载基础模型base_model可以指定模型的数据类型和是否信任远程代码
    # 根据参数选择加载到CPU或自动选择设备
    if args.device == 'auto':
        device_arg = {'device_map': 'auto'}
    else:
        device_arg = {'device_map': {"": args.device}}
    print(f"Loading base model: {args.base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float32 if args.device == "cpu" else torch.float16,
        trust_remote_code=args.trust_remote_code,
        **device_arg
    )

    # 循环加载并合并PEFT模型：遍历PEFT模型路径列表，加载每个PEFT模型，并将其合并到基础模型上。每次合并后，都会调用merge_and_unload方法释放资源
    model = base_model
    for peft_model_path in args.peft_model_paths:
        print(f"Loading PEFT: {peft_model_path}")
        model = PeftModel.from_pretrained(model, peft_model_path, **device_arg)
        print(f"Running merge_and_unload for {peft_model_path}")
        model = model.merge_and_unload()

    # Step3、加载对应分词器
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    # Step4、保存或发布合并后的模型
    # 如果指定了 --push_to_hub 参数，则将模型和 tokenizer 推送到 Hugging Face Hub。否则，调用 save_pretrained() 方法将模型和 tokenizer 保存到指定的输出目录。
    if args.push_to_hub:
        print(f"Saving to hub ...")
        model.push_to_hub(f"{args.output_dir}", use_temp_dir=False)
        tokenizer.push_to_hub(f"{args.output_dir}", use_temp_dir=False)
    else:
        model.save_pretrained(f"{args.output_dir}")
        tokenizer.save_pretrained(f"{args.output_dir}")
    # 输出保存模型的信息
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
