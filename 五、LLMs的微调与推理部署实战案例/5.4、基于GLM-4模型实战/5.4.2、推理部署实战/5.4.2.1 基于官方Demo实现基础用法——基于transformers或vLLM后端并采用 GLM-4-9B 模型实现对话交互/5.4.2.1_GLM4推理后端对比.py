# 文件用途：演示GLM-4基础对话
# 对应章节/小节：第5章 5.4.2.1 基于官方Demo实现基础用法——基于transformers或vLLM后端并采用 GLM-4-9B 模型实现对话交互
# 作者：NYY
# 创建日期：2024-10-12（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""分别用 Transformers 与 vLLM 运行同一个 GLM-4 问题。

参考：https://github.com/THUDM/GLM-4/tree/main/basic_demo
"""

import argparse

MODEL_PATH = "THUDM/glm-4-9b-chat"
MESSAGES = [{"role": "user", "content": "你好"}]


# T1、Transformers 后端
def run_t1():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Step1、加载模型和分词器，按模板编码消息
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype="auto", device_map="auto", trust_remote_code=True
    ).eval()
    inputs = tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(model.device)

    # Step2、生成新词元，并仅解码回复部分
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=512, do_sample=True,
                                   temperature=0.9, top_p=0.8)
    print(tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))


# T2、vLLM 后端
def run_t2():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # Step3、建立推理引擎，并用同一个模板组织输入
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    # 短问题演示只需有限上下文；显存需求同时受模型和 KV 缓存影响。
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=1, max_model_len=8192,
              trust_remote_code=True, enforce_eager=True)

    # Step4、设置 GLM-4 终止 token，生成并显示答案
    sampling = SamplingParams(temperature=0.9, top_p=0.8, max_tokens=512,
                              stop_token_ids=[151329, 151336, 151338])
    outputs = llm.generate([prompt], sampling)
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["T1", "T2"], default="T1",
                        help="T1 Transformers；T2 vLLM")
    args = parser.parse_args()
    {"T1": run_t1, "T2": run_t2}[args.method]()
