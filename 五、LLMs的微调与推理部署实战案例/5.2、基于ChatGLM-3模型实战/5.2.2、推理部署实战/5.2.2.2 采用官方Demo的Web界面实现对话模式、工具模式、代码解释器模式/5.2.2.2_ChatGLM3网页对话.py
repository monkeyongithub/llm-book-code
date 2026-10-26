# 文件用途：演示ChatGLM-3网页交互
# 对应章节/小节：第5章 5.2.2.2 采用官方Demo的Web界面实现对话模式、工具模式、代码解释器模式
# 作者：NYY
# 创建日期：2024-10-11（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""选择 Gradio 或 Streamlit 展示基础多轮对话。

工具模式和代码解释器模式由书稿所述 ChatGLM3/composite_demo 项目提供。
参考：https://github.com/THUDM/ChatGLM3/tree/main/basic_demo
"""

import argparse
import os
from threading import Lock


def load_model():
    from transformers import AutoModel, AutoTokenizer
    model_path = os.environ.get("MODEL_PATH", "THUDM/chatglm3-6b")
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("TOKENIZER_PATH", model_path),
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto").eval()
    return tokenizer, model, Lock()


# T1、Gradio：由 ChatInterface 管理每个页面的历史。
def run_gradio():
    import gradio as gr
    # Step1、加载一次模型，定义流式回复函数
    tokenizer, model, lock = load_model()
    def respond(message, history):
        messages = [{"role": item["role"], "content": item["content"]} for item in history]
        with lock:
            for response, _ in model.stream_chat(tokenizer, message, history=messages,
                                                  temperature=0.6, top_p=0.8):
                yield response
    # Step2、连接网页事件并启动服务
    demo = gr.ChatInterface(respond, type="messages", title="ChatGLM-3 对话示例")
    demo.queue().launch(server_name="127.0.0.1", server_port=7870, share=False)


# T2、Streamlit：用 session_state 保存当前会话的历史。
def run_streamlit():
    import streamlit as st
    # Step3、缓存模型，并初始化当前浏览器会话
    st.title("ChatGLM-3 对话示例")
    tokenizer, model, lock = st.cache_resource(load_model)()
    if "history" not in st.session_state:
        st.session_state.history = []
    if st.sidebar.button("清空历史"):
        st.session_state.history = []
    for message in st.session_state.history:
        if message["role"] in ("user", "assistant"):
            st.chat_message(message["role"]).write(message["content"])
    # Step4、接收输入并流式更新页面与会话记录
    if prompt := st.chat_input("请输入问题"):
        st.chat_message("user").write(prompt)
        placeholder = st.chat_message("assistant").empty()
        with lock:
            for response, history in model.stream_chat(tokenizer, prompt, history=st.session_state.history,
                                                       temperature=0.6, top_p=0.8):
                placeholder.write(response)
                st.session_state.history = history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ui", choices=["gradio", "streamlit"], default="gradio")
    args = parser.parse_args()
    if args.ui == "gradio":
        run_gradio()
    else:
        run_streamlit()
# Gradio：python <本文件相对路径> --ui gradio
# Streamlit：streamlit run <本文件相对路径> -- --ui streamlit
