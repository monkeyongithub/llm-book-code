# 文件用途：演示LLaMA-3 FastAPI服务
# 对应章节/小节：第5章 5.1.2.3 采用FastApi实现部署调用Llama-3-8B-Instruct
# 作者：NYY
# 创建日期：2024-10-10（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""用 FastAPI 演示请求校验、聊天模板及模型推理。

客户端提交自己的 history；服务不共享用户对话。服务启动后在另一进程调用 get_completion。
参考：https://github.com/datawhalechina/self-llm/tree/master/models/LLaMA3
"""

from contextlib import asynccontextmanager
from datetime import datetime
from threading import Lock
from typing import Literal

import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


# Step1、服务启动时加载模型，关闭时释放引用
@asynccontextmanager
async def lifespan(app):
    model_path = "models/LLM-Research/Meta-Llama-3-8B-Instruct"
    app.state.tokenizer = AutoTokenizer.from_pretrained(model_path)
    app.state.model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto"
    ).eval()
    app.state.generation_lock = Lock()
    yield
    del app.state.model, app.state.tokenizer


app = FastAPI(lifespan=lifespan)


# Step2、定义请求中的角色与消息内容
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    history: list[Message] = Field(default_factory=list)


# Step3、将本次请求的历史编码为模型输入并生成回复
@app.post("/")
def chat(payload: ChatRequest, request: Request):
    tokenizer, model = request.app.state.tokenizer, request.app.state.model
    messages = [message.model_dump() for message in payload.history]
    messages.append({"role": "user", "content": payload.prompt})
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with request.app.state.generation_lock, torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids, max_new_tokens=512, do_sample=True, temperature=0.6,
            top_p=0.9, pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        )
    response = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": response})
    return {"response": response, "history": messages, "status": 200,
            "time": datetime.now().isoformat(timespec="seconds")}


# Step4、客户端发送 JSON 并检查 HTTP 状态
def get_completion(prompt, history=None):
    import requests
    response = requests.post("http://127.0.0.1:6006/",
                             json={"prompt": prompt, "history": history or []}, timeout=120)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6006, workers=1)
