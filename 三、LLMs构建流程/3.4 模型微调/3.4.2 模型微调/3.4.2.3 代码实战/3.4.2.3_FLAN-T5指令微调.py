# 文件用途：演示FLAN-T5指令微调
# 对应章节/小节：第3章 3.4.2.3 代码实战
# 作者：NYY
# 创建日期：2024-10-27（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""在同一 DialogSum 样本上比较原始 FLAN-T5、全参数微调和 LoRA。

默认仅训练少量步骤以展示流程，ROUGE 的变化由实际运行结果决定。
各方案分别从同一个预训练检查点加载，评估使用独立的测试集。
"""

import argparse
import gc
from pathlib import Path

import evaluate
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
    TrainingArguments, set_seed,
)

MODEL_NAME = "google/flan-t5-base"
OUTPUT_DIR = Path("output/flan_t5_sft")


def make_prompt(dialogue):
    return f"Summarize the following conversation.\n\n{dialogue}\n\nSummary: "


@torch.inference_mode()
def summarize(model, tokenizer, dialogues):
    model.eval()
    summaries = []
    for dialogue in dialogues:
        inputs = tokenizer(make_prompt(dialogue), max_length=512, truncation=True,
                           return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        summaries.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return summaries


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=128)
    args = parser.parse_args()
    if args.max_steps <= 0 or args.train_samples <= 0:
        parser.error("训练步数和样本数必须大于零")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step1、读取数据，将输入对话和目标摘要分别编码
    dataset = load_dataset("knkarthick/dialogsum")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    training_data = dataset["train"].shuffle(seed=42).select(
        range(min(args.train_samples, len(dataset["train"])))
    )
    def tokenize(batch):
        features = tokenizer([make_prompt(text) for text in batch["dialogue"]],
                             max_length=512, truncation=True)
        features["labels"] = tokenizer(text_target=batch["summary"], max_length=128,
                                        truncation=True)["input_ids"]
        return features
    training_data = training_data.map(tokenize, batched=True, remove_columns=training_data.column_names)
    # DataCollatorForSeq2Seq 将标签填充为 -100，填充位置不计入损失。
    examples = dataset["test"].select(range(min(10, len(dataset["test"]))))
    dialogues, references = examples["dialogue"], examples["summary"]
    predictions = {}

    # Step2、记录未经微调的基线输出，再释放模型
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    predictions["原始模型"] = summarize(model, tokenizer, dialogues)
    del model
    release_memory()

    # Step3、分别微调独立模型，并保存本次训练得到的权重
    # T1、全参数微调：更新所有模型参数。
    # T2、LoRA 微调：冻结基础参数，只训练 q、v 投影的低秩适配器。
    for method, label in [("full", "全参数微调"), ("lora", "LoRA微调")]:
        set_seed(42)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        if method == "lora":
            model = get_peft_model(model, LoraConfig(
                r=32, lora_alpha=32, target_modules=["q", "v"],
                lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
            ))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"{label}：可训练参数 {trainable}/{total}")
        destination = OUTPUT_DIR / method
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(destination), max_steps=args.max_steps,
                per_device_train_batch_size=1, gradient_accumulation_steps=4,
                learning_rate=1e-5 if method == "full" else 1e-3,
                logging_steps=1, save_strategy="no", report_to="none", seed=42,
            ),
            train_dataset=training_data,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        )
        trainer.train()
        trainer.save_model(str(destination))
        tokenizer.save_pretrained(destination)
        predictions[label] = summarize(model, tokenizer, dialogues)
        del trainer, model
        release_memory()

    # Step4、在相同测试样本上比较摘要与 ROUGE，变化量以百分点表示
    rouge = evaluate.load("rouge")
    scores = {label: rouge.compute(predictions=texts, references=references, use_stemmer=True)
              for label, texts in predictions.items()}
    for index, reference in enumerate(references):
        print(f"\n样本 {index + 1}，参考摘要：{reference}")
        for label, texts in predictions.items():
            print(f"{label}：{texts[index]}")
    for label, result in scores.items():
        print(label, result)
        if label != "原始模型":
            print("相对原始模型的变化（百分点）：",
                  {key: round(100 * (value - scores["原始模型"][key]), 3)
                   for key, value in result.items()})
    # 少量样本用于观察流程，不能作为模型总体能力或微调收益的结论。


if __name__ == "__main__":
    main()
