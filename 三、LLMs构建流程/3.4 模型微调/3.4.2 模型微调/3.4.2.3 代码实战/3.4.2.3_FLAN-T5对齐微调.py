# 文件用途：演示FLAN-T5对齐微调
# 对应章节/小节：第3章 3.4.2.3 代码实战
# 作者：NYY
# 创建日期：2024-10-27（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""通过 RoBERTa 毒性奖励演示 FLAN-T5 的 PPO 对齐流程。

先运行同节指令微调示例，得到 output/flan_t5_sft/lora 适配器。
本例沿用书稿的 TRL PPOTrainer.step 接口；毒性分类分数不等同于整体摘要质量。
来源：https://github.com/Ryota-Kawamura/Generative-AI-with-LLMs/blob/main/Week-3/Lab_3_fine_tune_model_to_detoxify_summaries.ipynb
"""

from pathlib import Path
from statistics import mean, pstdev

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, set_seed
from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model

MODEL_NAME = "google/flan-t5-base"
ADAPTER_DIR = Path("output/flan_t5_sft/lora")
REWARD_MODEL = "facebook/roberta-hate-speech-dynabench-r4-target"


def collate(samples):
    return {key: [sample[key] for sample in samples] for key in samples[0]}


def label_score(scores, label):
    # pipeline 会按分数排序，因此应按标签查找，不能依赖列表下标。
    for item in scores:
        if item["label"].lower() == label:
            return float(item["score"])
    raise ValueError(f"奖励模型未返回 {label!r} 标签")


@torch.inference_mode()
def evaluate_summaries(model, tokenizer, samples, reward_pipeline):
    model.eval()
    summaries = []
    for sample in samples:
        inputs = sample["input_ids"].unsqueeze(0).to(next(model.parameters()).device)
        generated = model.generate(input_ids=inputs, max_new_tokens=128, do_sample=False)
        summaries.append(tokenizer.decode(generated[0], skip_special_tokens=True))
    if not summaries:
        raise ValueError("评估集不能为空")
    scores = reward_pipeline(summaries, top_k=None, function_to_apply="softmax", truncation=True)
    toxicities = [label_score(items, "hate") for items in scores]
    # 只评价生成的摘要，避免将原对话中的毒性计入模型输出。
    return summaries, mean(toxicities), pstdev(toxicities)


def main():
    set_seed(42)
    if not (ADAPTER_DIR / "adapter_config.json").is_file():
        raise FileNotFoundError("请先完成同节 FLAN-T5 指令微调示例，生成 LoRA 适配器")

    # Step1、筛选对话并构造摘要提示，分离训练与评估样本
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset("knkarthick/dialogsum", split="train")
    dataset = dataset.filter(lambda sample: 200 < len(sample["dialogue"]) <= 1000)
    def encode(sample):
        query = f"Summarize the following conversation.\n{sample['dialogue']}\nSummary:"
        return {"input_ids": tokenizer.encode(query, max_length=512, truncation=True), "query": query}
    dataset = dataset.map(encode, remove_columns=dataset.column_names)
    dataset.set_format(type="torch")
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    evaluation_samples = list(splits["test"].select(range(min(10, len(splits["test"])))))

    # Step2、建立待训练模型、冻结的参考模型及毒性奖励模型
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    adapted = PeftModel.from_pretrained(base, str(ADAPTER_DIR), is_trainable=True)
    policy = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(adapted)
    reference = create_reference_model(policy)
    reward_pipeline = pipeline("text-classification", model=REWARD_MODEL,
                               device=0 if torch.cuda.is_available() else -1)
    trainer = PPOTrainer(
        config=PPOConfig(model_name=MODEL_NAME, learning_rate=1.41e-5,
                         ppo_epochs=1, mini_batch_size=4, batch_size=16),
        model=policy, ref_model=reference, tokenizer=tokenizer,
        dataset=splits["train"], data_collator=collate,
    )
    before, mean_before, std_before = evaluate_summaries(policy, tokenizer, evaluation_samples, reward_pipeline)

    # Step3、采样摘要，以“非仇恨”原始 logits 为奖励执行少量 PPO 更新
    policy.train()
    for step, batch in enumerate(trainer.dataloader):
        if step >= 10:
            break
        queries = [ids.to(trainer.accelerator.device) for ids in batch["input_ids"]]
        responses = []
        for query in queries:
            response = trainer.generate(query, max_new_tokens=128, min_new_tokens=5,
                                        do_sample=True, top_k=0, top_p=1.0)
            # T5 的输出已经是解码器序列，无需减去编码器输入长度。
            responses.append(response.squeeze(0))
        batch["response"] = [tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
        scores = reward_pipeline(batch["response"], top_k=None,
                                 function_to_apply="none", truncation=True)
        rewards = [torch.tensor(label_score(items, "nothate"), device=trainer.accelerator.device)
                   for items in scores]
        stats = trainer.step(queries, responses, rewards)
        trainer.log_stats(stats, batch, rewards)
        print(f"步骤 {step + 1}，KL={stats['objective/kl']}")

    # Step4、在相同样本上比较摘要毒性，并保留文本供读者检查忠实性
    after, mean_after, std_after = evaluate_summaries(policy, tokenizer, evaluation_samples, reward_pipeline)
    print(f"微调前：均值={mean_before:.4f}，标准差={std_before:.4f}")
    print(f"微调后：均值={mean_after:.4f}，标准差={std_after:.4f}")
    if mean_before > 0:
        print(f"毒性均值相对下降：{(mean_before - mean_after) / mean_before:.2%}")
    for original, aligned in zip(before, after):
        print(f"\n微调前：{original}\n微调后：{aligned}")
    # 分数下降也可能来自遗漏内容，须结合摘要完整性与事实一致性一起判断。


if __name__ == "__main__":
    main()
