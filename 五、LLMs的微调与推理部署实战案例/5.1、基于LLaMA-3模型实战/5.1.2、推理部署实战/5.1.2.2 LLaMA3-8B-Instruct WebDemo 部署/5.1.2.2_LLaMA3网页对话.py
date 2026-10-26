# 文件用途：演示LLaMA-3网页对话
# 对应章节/小节：第5章 5.1.2.2 LLaMA3-8B-Instruct WebDemo 部署
# 作者：NYY
# 创建日期：2024-10-10（文件系统记录）
# 运行方式：从代码根目录使用 streamlit run 加本文件相对路径启动
# 备注：核心演示代码

"""用 Streamlit 展示聊天模板、生成回复与会话状态。

参考：https://github.com/datawhalechina/self-llm/tree/master/models/LLaMA3
"""

from threading import Lock

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Step1、缓存模型与推理锁，页面刷新时复用模型
@st.cache_resource
def get_model():
    model_path = "models/LLM-Research/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto"
    ).eval()
    return tokenizer, model, Lock()


st.title("LLaMA-3 对话示例")
tokenizer, model, generation_lock = get_model()

# Step2、每个浏览器会话维护独立的角色消息
if "messages" not in st.session_state:
    st.session_state.messages = []
if st.sidebar.button("清空对话"):
    st.session_state.messages = []
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# Step3、用分词器模板组织完整历史，并生成 assistant 回复
if prompt := st.chat_input("请输入问题"):
    messages = [*st.session_state.messages, {"role": "user", "content": prompt}]
    st.chat_message("user").write(prompt)
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with generation_lock, torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids, max_new_tokens=512, do_sample=True,
            temperature=0.6, top_p=0.9,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)

    # Step4、成功生成后写回历史，并显示本轮回复
    st.session_state.messages = messages + [{"role": "assistant", "content": response}]
    st.chat_message("assistant").write(response)
