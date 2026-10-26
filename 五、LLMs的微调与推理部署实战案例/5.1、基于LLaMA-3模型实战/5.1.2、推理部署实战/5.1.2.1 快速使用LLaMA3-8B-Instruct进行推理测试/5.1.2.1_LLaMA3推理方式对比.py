# 文件用途：演示LLaMA-3推理
# 对应章节/小节：第5章 5.1.2.1 快速使用LLaMA3-8B-Instruct进行推理测试
# 作者：NYY
# 创建日期：2024-10-10（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

import argparse

def run_t1():
    # T1、基于 Transformers 直接生成
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    # Step1、加载模型并准备分词器
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Step2、按聊天模板组织输入
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Step3、直接生成并解码回复
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))


def run_t2():
    # T2、基于 Transformers pipeline 生成
    import transformers
    import torch
    torch.cuda.empty_cache()


    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Step4、创建 Transformers 生成管道
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    # Step5、构造管道输入
    instruction = "You are a pirate chatbot who always responds in pirate speak!"
    user_input = "Who are you?"

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Step6、从管道输出中提取回复
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    print(outputs[0]["generated_text"][len(prompt):])



def run_t3():
    # T3、基于 LangChain 调用 NVIDIA 接口
    # 来源：https://python.langchain.com/v0.2/docs/integrations/chat/nvidia_ai_endpoints/
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    # Step7、定义 LangChain 提示与调用链
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful AI assistant named Fred."), ("user", "{input}")]
    )
    chain = prompt | ChatNVIDIA(model="meta/llama3-8b-instruct") | StrOutputParser()

    # Step8、流式展示在线接口回复
    for txt in chain.stream({"input": "What's your name?"}):
        print(txt, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="每次选择一种 LLaMA-3 推理方案")
    parser.add_argument("--method", choices=["T1", "T2", "T3"], default="T1",
                        help="T1 直接生成；T2 pipeline；T3 NVIDIA 在线接口")
    args = parser.parse_args()
    {"T1": run_t1, "T2": run_t2, "T3": run_t3}[args.method]()
