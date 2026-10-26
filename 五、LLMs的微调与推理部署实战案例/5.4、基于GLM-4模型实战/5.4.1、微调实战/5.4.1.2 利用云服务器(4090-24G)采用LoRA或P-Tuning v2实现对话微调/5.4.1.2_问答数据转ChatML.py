# -*- coding: utf-8 -*-
# 文件用途：将问答文本转换为ChatML数据
# 对应章节/小节：第5章 5.4.1.2 利用云服务器(4090-24G)采用LoRA或P-Tuning v2实现对话微调
# 作者：NYY
# 创建日期：2024-10-13（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""将以“问：”“答：”开头的文本转换为 ChatML JSONL，保留多行内容。"""

import argparse
import json
from pathlib import Path


def convert_txt_to_chatml(input_file, output_file):
    source, destination = Path(input_file), Path(output_file)
    if source.resolve() == destination.resolve():
        raise ValueError("输入和输出不能是同一个文件")

    # Step1、按问答标记归组，保留后续行并拒绝不完整样本
    pairs, question, answer, section = [], [], [], None
    def finish_pair():
        q, a = "\n".join(question).strip(), "\n".join(answer).strip()
        if not q or not a:
            raise ValueError(f"第 {len(pairs) + 1} 组问答内容不完整")
        pairs.append((q, a))
    for number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if line.startswith("问："):
            if section is not None:
                finish_pair()
            question, answer, section = [line[2:].strip()], [], "question"
        elif line.startswith("答："):
            if section != "question":
                raise ValueError(f"第 {number} 行的回答缺少对应问题或出现重复标记")
            answer, section = [line[2:].strip()], "answer"
        elif section is not None:
            (question if section == "question" else answer).append(line)
        elif line:
            raise ValueError(f"第 {number} 行应以“问：”开始")
    if section is not None:
        finish_pair()
    if not pairs:
        raise ValueError("没有找到问答样本")

    # Step2、全部解析成功后再写出文件，避免格式错误时截断输出
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for question_text, answer_text in pairs:
            sample = {"messages": [{"role": "user", "content": question_text},
                                   {"role": "assistant", "content": answer_text}]}
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return len(pairs)


# Step3、从命令行接收相对路径并报告转换数量
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file")
    args = parser.parse_args()
    source = Path(args.input_file)
    destination = Path(args.output_file) if args.output_file else source.with_name(source.stem + "_chatml.jsonl")
    count = convert_txt_to_chatml(source, destination)
    print(f"已转换 {count} 组问答，保存到 {destination}")
