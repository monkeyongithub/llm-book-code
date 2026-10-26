# 文件用途：说明Colab微调流程
# 对应章节/小节：第5章 5.1.1.2 基于Colab平台采用LLaMA-Factory工具和unsloth优化框架实现LoRA微调llama-3-8b-Instruct
# 作者：NYY
# 创建日期：2024-10-11（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""按 prepare、chat、export-config 三个阶段阅读 Colab 微调流程。

准备配置后，使用 LLaMA-Factory CLI 完成训练，再选择 chat 测试适配器。
参考：https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9
"""

import argparse
import json
import re
from pathlib import Path

OUTPUT = Path("output/llama3_factory")
MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"
ADAPTER = OUTPUT / "lora"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare():
    # Step1、读取身份数据，替换占位符并写到独立的输出目录
    source = Path("LLaMA-Factory/data")
    dataset = json.loads((source / "identity.json").read_text(encoding="utf-8"))
    replacements = {"name": "Llama-3", "author": "NYY"}
    for sample in dataset:
        for key, value in replacements.items():
            sample["output"] = re.sub(r"\{\{\s*" + key + r"\s*\}\}", value, sample["output"])
    save_json(OUTPUT / "data/identity.json", dataset)
    # alpaca_gpt4_en 与数据集登记文件沿用 LLaMA-Factory 的格式。
    alpaca = json.loads((source / "alpaca_gpt4_data_en.json").read_text(encoding="utf-8"))
    save_json(OUTPUT / "data/alpaca_gpt4_data_en.json", alpaca)
    save_json(OUTPUT / "data/dataset_info.json", {
        "identity": {"file_name": "identity.json"},
        "alpaca_gpt4_en": {"file_name": "alpaca_gpt4_data_en.json"},
    })

    # Step2、生成 CLI 训练配置；写出配置不等同于已经训练模型
    config = dict(
        stage="sft", do_train=True, model_name_or_path=MODEL,
        dataset_dir=str(OUTPUT / "data"), dataset="identity,alpaca_gpt4_en",
        template="llama3", finetuning_type="lora", lora_target="all",
        output_dir=str(ADAPTER), per_device_train_batch_size=2,
        gradient_accumulation_steps=4, lr_scheduler_type="cosine", logging_steps=10,
        warmup_ratio=0.1, save_steps=1000, learning_rate=5e-5, num_train_epochs=3.0,
        max_samples=500, max_grad_norm=1.0, quantization_bit=4, loraplus_lr_ratio=16.0,
        use_unsloth=True, fp16=True,
    )
    save_json(OUTPUT / "train.json", config)
    # T1、图形界面：运行 llamafactory-cli webui 后选择相同数据与参数。
    # T2、命令行：运行以下命令训练，再选择本文件的 chat 阶段。
    print(f"llamafactory-cli train {OUTPUT / 'train.json'}")


def chat():
    # Step3、检查适配器已生成，再加载模型开始多轮对话
    if not (ADAPTER / "adapter_config.json").is_file():
        raise FileNotFoundError("请先使用 prepare 生成的配置完成训练")
    from llamafactory.chat import ChatModel
    model = ChatModel(dict(model_name_or_path=MODEL, adapter_name_or_path=str(ADAPTER),
                           template="llama3", finetuning_type="lora", quantization_bit=4,
                           use_unsloth=True))
    messages = []
    while True:
        try:
            query = input("User（clear 清空，exit 退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query == "exit":
            break
        if query == "clear":
            messages.clear()
            continue
        if not query:
            continue
        messages.append({"role": "user", "content": query})
        response = ""
        for text in model.stream_chat(messages):
            print(text, end="", flush=True)
            response += text
        print()
        messages.append({"role": "assistant", "content": response})


def export_config():
    # Step4、使用非量化基础模型配置合并导出
    if not (ADAPTER / "adapter_config.json").is_file():
        raise FileNotFoundError("请先完成训练再导出")
    config = dict(model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
                  adapter_name_or_path=str(ADAPTER), template="llama3", finetuning_type="lora",
                  export_dir=str(OUTPUT / "merged"), export_size=2, export_device="cpu")
    save_json(OUTPUT / "export.json", config)
    print(f"llamafactory-cli export {OUTPUT / 'export.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "chat", "export-config"], default="prepare")
    args = parser.parse_args()
    {"prepare": prepare, "chat": chat, "export-config": export_config}[args.stage]()
