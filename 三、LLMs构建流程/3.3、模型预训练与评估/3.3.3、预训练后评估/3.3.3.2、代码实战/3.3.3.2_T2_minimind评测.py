# 文件用途：演示C-Eval自动评估
# 对应章节/小节：第3章 3.3.3.2 代码实战：基准评估案例（C-Evals）
# 作者：NYY
# 创建日期：2024-10-27（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""minimind 的 C-Eval 评估核心流程。

需配合书稿所用 minimind 项目的 model.model、model.LMConfig 与权重使用。
本例用单个选项 token 的下一词概率评分；不适用于选项被分成多个 token 的分词器。
来源：https://github.com/jingyaogong/minimind/blob/master/eval_ceval.py
"""

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer
from model.model import Transformer
from model.LMConfig import LMConfig


def load_model(device):
    config = LMConfig()
    tokenizer = AutoTokenizer.from_pretrained(
        "model/minimind_tokenizer", trust_remote_code=True, use_fast=False
    )
    suffix = "_moe" if config.use_moe else ""
    checkpoint = f"out/single_chat/full_sft_{config.dim}{suffix}.pth"
    model = Transformer(config)
    state = torch.load(checkpoint, map_location="cpu")
    state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval(), tokenizer


@torch.inference_mode()
def predict_option(model, tokenizer, row, option_ids, device):
    question = str(row["question"]) + "\n" + "\n".join(
        f"{option}. {row[option]}" for option in "ABCD"
    )
    messages = [
        {"role": "system", "content": "回答选择题，只输出一个选项字母 A、B、C 或 D。"},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    logits = model.eval_answer(inputs)
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("eval_answer 应返回 [1, vocab_size] 或 [1, seq_len, vocab_size] 的 logits")
    # 对四个候选的 logits 归一化，得到在四个选项之间的相对概率。
    probabilities = torch.softmax(logits[0, option_ids].float(), dim=-1)
    return "ABCD"[probabilities.argmax().item()]


def main():
    # Step1、加载模型，并检查每个选项是否对应单个 token
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(device)
    encoded_options = [tokenizer.encode(option, add_special_tokens=False) for option in "ABCD"]
    if any(len(ids) != 1 for ids in encoded_options):
        raise ValueError("该分词器的选项不是单 token，应使用完整选项序列的条件概率评分")
    option_ids = [ids[0] for ids in encoded_options]
    if len(set(option_ids)) != 4:
        raise ValueError("四个选项必须对应不同的 token")

    # Step2、读取各科目题目，逐题预测并记录结果
    input_dir = Path("ceval/ceval-exam/val")
    output_dir = Path("ceval/ceval_result")
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"没有找到评测文件：{input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    total_correct = total_questions = 0
    summaries = []
    for path in files:
        frame = pd.read_csv(path)
        required = {"question", "A", "B", "C", "D", "answer"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path.name} 缺少列：{sorted(required - set(frame.columns))}")
        rows = []
        for _, row in frame.iterrows():
            expected = str(row["answer"]).strip().upper()
            if expected not in "ABCD" or len(expected) != 1:
                raise ValueError(f"{path.name} 存在非法答案：{expected!r}")
            predicted = predict_option(model, tokenizer, row, option_ids, device)
            rows.append({**row.to_dict(), "llm_answer": predicted, "is_right": int(predicted == expected)})
        results = pd.DataFrame(rows, columns=[*frame.columns, "llm_answer", "is_right"])
        results.to_csv(output_dir / f"{path.stem}_result.csv", index=False)
        correct = sum(row["is_right"] for row in rows)
        total_correct += correct
        total_questions += len(rows)
        accuracy = correct / len(rows) if rows else 0.0
        summaries.append(f"{path.stem}: {correct}/{len(rows)}，准确率 {accuracy:.2%}")

    # Step3、按题目数汇总准确率，避免将科目准确率简单平均
    overall = total_correct / total_questions if total_questions else 0.0
    summaries.append(f"总计：{total_correct}/{total_questions}，准确率 {overall:.2%}")
    report = "\n".join(summaries)
    (output_dir / "test.log").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
