# 文件用途：演示Qwen工具调用
# 对应章节/小节：第8章 8.3.2.3 基于Qwen官方案例实现工具调用能力
# 作者：NYY
# 创建日期：2024-10-30（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""Qwen 多轮工具调用：真实本地时间与明确标注的模拟天气。"""

import json
import os
from datetime import datetime

from openai import OpenAI


# Step1、定义函数与对应 JSON Schema
def get_current_weather(location):
    if not isinstance(location, str) or not location.strip():
        raise ValueError("location 必须为非空字符串")
    return f"模拟天气：{location}为雨天；这不是实时查询结果。"


def get_current_time():
    return datetime.now().astimezone().isoformat(timespec="seconds")


TOOLS = [
    {"type": "function", "function": {
        "name": "get_current_time", "description": "获取运行计算机当前的本地时间及 UTC 偏移量",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }},
    {"type": "function", "function": {
        "name": "get_current_weather", "description": "返回指定地点的模拟天气，仅演示工具流程",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string", "description": "城市或县区"}},
            "required": ["location"], "additionalProperties": False},
    }},
]


def main():
    # Step2、创建客户端，并提交用户问题和可用工具
    client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    messages = [{"role": "user", "content": input("请输入问题：")}]
    functions = {"get_current_time": get_current_time, "get_current_weather": get_current_weather}
    for _ in range(8):
        reply = client.chat.completions.create(model="qwen-max", messages=messages, tools=TOOLS).choices[0].message
        messages.append(reply.model_dump(exclude_none=True))
        if not reply.tool_calls:
            print(reply.content or "")
            return

        # Step3、执行全部调用；arguments 是参数对象，不含 Schema 的 properties 层
        for call in reply.tool_calls:
            try:
                if call.function.name not in functions:
                    raise ValueError("未知工具")
                arguments = json.loads(call.function.arguments)
                result = functions[call.function.name](**arguments)
            except (ValueError, TypeError) as error:
                result = f"工具参数错误：{error}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        # 交回同一消息列表，让模型使用工具结果继续回答。
    raise RuntimeError("工具调用轮数超过示例上限")


if __name__ == "__main__":
    main()
