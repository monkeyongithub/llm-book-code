# 文件用途：说明GLM-4服务部署
# 对应章节/小节：第5章 5.4.2.3 基于云服务器(4090-24G)实现一键部署开启服务并交互测试
# 作者：NYY
# 创建日期：2024-10-13（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""Flask 网页对话示例：浏览器携带本会话历史，服务只共享模型。

模型路径可指向 GLM-4 基础模型或配套微调产生的 PEFT 适配器目录。
"""

import os
from functools import lru_cache
from pathlib import Path
from threading import Lock

import torch
from flask import Flask, jsonify, render_template, request
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


# Step1、加载一次基础模型或 LoRA 模型，兼容本地路径和模型仓库标识
@lru_cache(maxsize=1)
def load_model():
    model_path = os.environ.get("MODEL_PATH", "GLM-4/finetune_demo/output/lora/checkpoint-200-identity")
    if (Path(model_path) / "adapter_config.json").is_file():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, device_map="auto"
        )
        tokenizer_path = model.peft_config["default"].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, device_map="auto"
        )
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True,
                                              encode_special_tokens=True, use_fast=False)
    return model.eval(), tokenizer


# Step2、提供配套 HTML 页面
app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
generation_lock = Lock()


@app.get("/")
def index():
    return render_template("5.4.2.3_GLM4聊天页面.html")


# Step3、校验本次请求，重建当前浏览器的对话消息
@app.post("/chat")
def chat():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="请求内容应为 JSON 对象"), 400
    message, history = payload.get("message"), payload.get("history", [])
    if not isinstance(message, str) or not message.strip():
        return jsonify(error="message 必须是非空字符串"), 400
    if not isinstance(history, list) or any(
        not isinstance(pair, list) or len(pair) != 2
        or not all(isinstance(text, str) for text in pair) for pair in history
    ):
        return jsonify(error="history 应为问答字符串对的列表"), 400
    messages = []
    for question, answer in history:
        messages.extend([{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}])
    messages.append({"role": "user", "content": message.strip()})

    # Step4、串行调用共享模型，再将新历史返回给当前浏览器
    # 页面在完整生成后更新，因此直接生成即可，无需额外的流式后台线程。
    with generation_lock, torch.inference_mode():
        model, tokenizer = load_model()
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        generated = model.generate(input_ids=input_ids, max_new_tokens=512,
                                   do_sample=True, top_p=0.7, temperature=0.9,
                                   eos_token_id=model.config.eos_token_id)
        response = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
    return jsonify(response=response, history=history + [[message.strip(), response]])


if __name__ == "__main__":
    load_model()
    app.run(host="127.0.0.1", port=6006)
