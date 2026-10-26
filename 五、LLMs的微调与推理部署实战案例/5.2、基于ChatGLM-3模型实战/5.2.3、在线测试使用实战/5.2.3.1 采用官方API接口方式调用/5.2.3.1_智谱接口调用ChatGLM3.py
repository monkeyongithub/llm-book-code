# -*- coding: utf-8 -*-
# 文件用途：演示ChatGLM-3在线API调用
# 对应章节/小节：第5章 5.2.3.1 采用官方API接口方式调用
# 作者：NYY
# 创建日期：2024-10-13（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# 采用官方API接口方式调用
# pip install zhipuai 请先在终端进行安装
import os

from zhipuai import ZhipuAI
api_key = os.environ["ZHIPUAI_API_KEY"]
# Step1、配置智谱客户端并提交对话请求
client = ZhipuAI(api_key=api_key) # 填写您自己的APIKey
response = client.chat.completions.create(
    model ="charglm-3",
    messages = [
        {
            "role": "user",
            "content": "你好，请进行自我介绍"
        }
    ],
    temperature= 0.9,
    top_p= 0.7,
    meta = {
        "bot_info": "苏轼，字子瞻，世人称呼他为苏东坡。苏轼是四川省眉山市人，也是北宋文学家，书法家、画家。苏轼的父亲为苏洵，弟弟为苏辙，父子三人并称“三苏”。苏轼是北宋中期文坛领袖，在诗、词、文、书、画等方面取得很高成就。其诗题材广阔，清新豪健，善用夸张比喻，独具风格，与黄庭坚并称“苏黄”。苏轼的词开豪放一派，与辛弃疾同是豪放派代表，并称“苏辛”。苏轼的文章著述宏富，纵横恣肆，豪放自如，与欧阳修并称“欧苏”。苏轼擅长文人画，尤其擅长墨竹、怪石、枯木等。苏轼的作品有《东坡七集》《东坡易传》《东坡乐府》《寒食帖》《潇湘竹石图》《枯木怪石图》等。",
        "bot_name": "苏东坡",
        "user_info": "用户",
        "user_name": "用户"
    },
    stream=True
)


# Step2、选择流式展示或打字机展示。
print_stream = False  # True：直接展示流；False：收集文本后逐字展示。
if print_stream:
    # T1、逐块展示接口流式输出
    for trunk in response:
        print((trunk.choices[0].delta.content or '') if trunk.choices else '', end='', flush=True)  # 使用end=''避免自动换行，flush=True确保即时输出


else:
    # T2、先收集流中的文本，再演示打字机效果
    full_text = ""
    for trunk in response:
        if trunk.choices:
            full_text += trunk.choices[0].delta.content or ""
    print(full_text)

    import time
    import sys
    for char in full_text:
        # 输出当前字符并刷新缓冲区
        sys.stdout.write(char)
        sys.stdout.flush()
        # 设置字符输出的延迟时间
        time.sleep(0.1)  # 可以根据需要调整这个值，例如 0.1 秒
    print()
