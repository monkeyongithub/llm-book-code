# 文件用途：演示ChatGLM-3命令行对话
# 对应章节/小节：第5章 5.2.2.1 采用官方Demo的CLI的形式实现对话交互
# 作者：NYY
# 创建日期：2024-10-11（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""加载一次 ChatGLM-3，演示多轮历史与流式输出。

来源：https://github.com/THUDM/ChatGLM3/tree/main/basic_demo
"""

import os

from transformers import AutoModel, AutoTokenizer


def main():
    # Step1、加载模型与分词器
    model_path = os.environ.get("MODEL_PATH", "THUDM/chatglm3-6b")
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("TOKENIZER_PATH", model_path),
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto").eval()
    history = []
    print("输入 clear 清空历史，stop 退出。")

    # Step2、读取问题并将历史交给模型
    while True:
        try:
            query = input("\n用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query == "stop":
            break
        if query == "clear":
            history = []
            print("历史已清空。")
            continue
        if not query:
            continue

        # Step3、只输出新增的流式文本，并保留模型返回的消息历史
        print("ChatGLM：", end="", flush=True)
        previous_length = 0
        for response, updated_history in model.stream_chat(tokenizer, query, history=history,
                                                           top_p=0.8, temperature=0.6):
            print(response[previous_length:], end="", flush=True)
            previous_length = len(response)
            history = updated_history
        print()


if __name__ == "__main__":
    main()
