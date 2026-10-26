# 文件用途：演示LangGraph工具调用工作流
# 对应章节/小节：第8章 8.3.2.5 基于LangGraph框架实现GraphETA
# 作者：NYY
# 创建日期：2024-10-30（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""用消息图连接模型与乘法工具，工具结果返回模型后才结束回答。"""

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, MessageGraph
from langgraph.prebuilt import ToolNode


# Step1、定义乘法工具并绑定到 Qwen
@tool
def multiply(first_int: int, second_int: int) -> int:
    """计算两个整数的乘积。"""
    return first_int * second_int


llm = ChatTongyi(model="qwen-max").bind_tools([multiply])

# Step2、构建模型节点、工具节点和条件路由
builder = MessageGraph()
builder.add_node("model", llm)
builder.add_node("tools", ToolNode([multiply]))
builder.set_entry_point("model")


def route(messages):
    return "tools" if getattr(messages[-1], "tool_calls", None) else END


builder.add_conditional_edges("model", route, {"tools": "tools", END: END})
builder.add_edge("tools", "model")
graph = builder.compile()

# Step3、分别观察需要工具与无需工具的路径
if __name__ == "__main__":
    for question in ["123 乘以 456 是多少？", "你叫什么名字？"]:
        messages = [SystemMessage(content="需要计算时调用工具，再根据工具结果回答。"),
                    HumanMessage(content=question)]
        result = graph.invoke(messages, config={"recursion_limit": 10})
        print(result[-1].content)
        # ToolNode 已执行函数，应用层无需再次计算或伪造 assistant 消息。
