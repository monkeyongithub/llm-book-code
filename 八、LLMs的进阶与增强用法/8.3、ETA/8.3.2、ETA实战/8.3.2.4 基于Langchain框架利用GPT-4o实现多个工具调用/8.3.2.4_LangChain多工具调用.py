# 文件用途：演示LangChain多工具调用
# 对应章节/小节：第8章 8.3.2.4 基于Langchain框架利用GPT-4o实现多个工具调用
# 作者：NYY
# 创建日期：2024-10-30（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""比较三种工具模式，再执行模型提出的加法与乘法调用。

模式负责说明参数，普通 Python 函数负责执行；两者不能相互替代。
参考：https://python.langchain.com/v0.2/docs/how_to/tool_calling/
"""

import argparse
import json

from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI


# Step1、定义执行函数及三种等价的参数模式
# T1、Python 函数：类型提示和文档字符串组成模式。
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b


def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""
    return a * b


# T2、Pydantic：声明参数约束，并可在执行前校验。
class AddSchema(BaseModel):
    """计算两个整数的和。"""
    a: int = Field(description="First integer", strict=True)
    b: int = Field(description="Second integer", strict=True)


class MultiplySchema(AddSchema):
    """计算两个整数的乘积。"""


# T3、TypedDict：声明字典参数结构。
class AddTypedDict(TypedDict):
    """计算两个整数的和。"""
    a: Annotated[int, ..., "First integer"]
    b: Annotated[int, ..., "Second integer"]


class MultiplyTypedDict(TypedDict):
    """计算两个整数的乘积。"""
    a: Annotated[int, ..., "First integer"]
    b: Annotated[int, ..., "Second integer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["T1", "T2", "T3"], default="T1")
    args = parser.parse_args()
    modes = {"T1": [add, multiply], "T2": [AddSchema, MultiplySchema],
             "T3": [AddTypedDict, MultiplyTypedDict]}
    schemas = []
    for name, definition in zip(("add", "multiply"), modes[args.method]):
        schema = convert_to_openai_tool(definition)
        schema["function"]["name"] = name
        schemas.append(schema)

    # Step2、绑定所选模式，让模型生成调用请求
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(schemas)
    messages = [HumanMessage(content="What is 3 * 12? Also, what is 11 + 49?")]
    functions = {"add": add, "multiply": multiply}
    for _ in range(8):
        reply = llm.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            print(reply.content)
            return

        # Step3、校验参数，执行白名单函数，将每个结果关联到调用 ID
        for call in reply.tool_calls:
            try:
                if call["name"] not in functions:
                    raise ValueError("未知工具")
                arguments = AddSchema.model_validate(call["args"]).model_dump()
                result = functions[call["name"]](**arguments)
            except (TypeError, ValueError) as error:
                result = {"error": str(error)}
            messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False),
                                        tool_call_id=call["id"]))
        # 下一轮模型可根据工具结果回答，或继续提出有依赖的工具调用。
    raise RuntimeError("工具调用轮数超过示例上限")


if __name__ == "__main__":
    main()
