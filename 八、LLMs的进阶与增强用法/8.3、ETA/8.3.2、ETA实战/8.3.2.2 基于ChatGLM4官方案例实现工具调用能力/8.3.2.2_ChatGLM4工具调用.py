# 文件用途：演示ChatGLM4工具调用
# 对应章节/小节：第8章 8.3.2.2 基于ChatGLM4官方案例实现工具调用能力
# 作者：NYY
# 创建日期：2024-10-30（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""用模拟航班数据演示 GLM-4 的参数生成、函数执行与多轮工具消息。"""

import json
import os
from datetime import date

from zhipuai import ZhipuAI


# Step1、定义模拟航班工具；数据只对应给定日期，不能用于真实出行查询
def get_flight_number(date: str, departure: str, destination: str):
    validate_date(date)
    flights = {("北京", "上海"): "1234", ("北京", "广州"): "8321",
               ("上海", "北京"): "1233", ("上海", "广州"): "8123"}
    return {"flight_number": flights.get((departure, destination)), "date": date, "simulated": True}


def get_ticket_price(date: str, flight_number: str):
    validate_date(date)
    price = 1000 if flight_number in {"1234", "8321", "1233", "8123"} else None
    return {"ticket_price": price, "currency": "CNY", "simulated": True}


def validate_date(value):
    date.fromisoformat(value)
    if value != "2024-01-23":
        raise ValueError("演示数据仅提供 2024-01-23 的模拟航班")


# Step2、用 Schema 向模型说明可调用函数
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_flight_number",
            "description": "根据始发地、目的地和日期，查询对应日期的航班号",
            "parameters": {
                "type": "object",
                "properties": {
                    "departure": {
                        "description": "出发地",
                        "type": "string"
                    },
                    "destination": {
                        "description": "目的地",
                        "type": "string"
                    },
                    "date": {
                        "description": "日期",
                        "type": "string",
                    }
                },
                "required": [ "departure", "destination", "date" ]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket_price",
            "description": "查询某航班在某日的票价",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_number": {
                        "description": "航班号",
                        "type": "string"
                    },
                    "date": {
                        "description": "日期",
                        "type": "string",
                    }
                },
                "required": [ "flight_number", "date"]
            },
        }
    },
]


def ask(client, messages, question):
    messages.append({"role": "user", "content": question})
    functions = {"get_flight_number": get_flight_number, "get_ticket_price": get_ticket_price}
    # Step3、处理有依赖的多轮调用，并逐一返回工具结果
    for _ in range(8):
        reply = client.chat.completions.create(model="glm-4", messages=messages, tools=tools).choices[0].message
        messages.append(reply.model_dump(exclude_none=True))
        if not reply.tool_calls:
            print(reply.content or "")
            return
        for call in reply.tool_calls:
            try:
                if call.function.name not in functions:
                    raise ValueError("未知工具")
                result = functions[call.function.name](**json.loads(call.function.arguments))
            except (ValueError, TypeError) as error:
                result = {"error": str(error)}
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})
    raise RuntimeError("工具调用轮数超过示例上限")


# Step4、先查询航班，再基于同一历史查询价格
if __name__ == "__main__":
    client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])
    messages = [{"role": "system", "content": "这是模拟航班查询。不要猜测参数，缺少信息时请用户补充。"}]
    ask(client, messages, "查询 2024 年 1 月 23 日北京到广州的航班。")
    ask(client, messages, "这趟航班的票价是多少？")
