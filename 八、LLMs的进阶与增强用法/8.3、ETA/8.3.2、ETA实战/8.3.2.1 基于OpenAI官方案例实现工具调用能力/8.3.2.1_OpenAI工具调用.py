# 文件用途：演示OpenAI工具调用
# 对应章节/小节：第8章 8.3.2.1 基于OpenAI官方案例实现工具调用能力
# 作者：NYY
# 创建日期：2024-10-30（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：订单查询使用模拟数据

"""展示工具定义、调用处理和将工具结果返回模型的流程。"""

import json
from datetime import date

from openai import OpenAI


# Step1、描述订单查询工具的参数
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_delivery_date",
        "description": "查询示例订单的配送日期",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "订单编号"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
}


# Step2、实现示例订单查询函数
def get_delivery_date(order_id: str) -> str:
    """返回模拟订单的日期；不连接真实订单系统。"""
    if order_id == "order_12345":
        return date.today().isoformat()
    return "该订单没有示例数据"


# Step3、让模型选择工具并回传执行结果
def main() -> None:
    client = OpenAI()
    messages = [
        {"role": "system", "content": "你是客服助手。订单查询结果仅供流程演示。"},
        {"role": "user", "content": "我的订单 order_12345 预计何时送达？"},
    ]
    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, tools=[TOOL_SCHEMA]
    )
    message = response.choices[0].message
    if not message.tool_calls:
        print(message.content or "")
        return

    messages.append(message.model_dump(exclude_none=True))
    for call in message.tool_calls:
        try:
            arguments = json.loads(call.function.arguments)
            if call.function.name != "get_delivery_date":
                result = "不支持的工具"
            else:
                result = get_delivery_date(**arguments)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            result = f"工具参数无效：{error}"
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": result,
        })

    final_response = client.chat.completions.create(
        model="gpt-4o", messages=messages
    )
    print(final_response.choices[0].message.content or "")


if __name__ == "__main__":
    main()
