# 文件用途：根据模拟请求记录统计延迟、错误率、吞吐量与 token 用量
# 对应章节/小节：第4章 4.2 模型监控与维护
# 作者：NYY
# 创建日期：2024-12-31
# 运行方式：从代码根目录运行本文件的相对路径
# 备注：配合书稿概念阅读的小型演示

"""演示固定观察窗口的服务统计；不连接真实服务或监控平台。"""

import math
from statistics import mean


# Step1、准备观察窗口内完成的请求记录，延迟单位为毫秒
requests = [
    {"latency_ms": 120, "ok": True, "tokens": 42},
    {"latency_ms": 180, "ok": True, "tokens": 60},
    {"latency_ms": 800, "ok": False, "tokens": 0},
    {"latency_ms": 150, "ok": True, "tokens": 38},
    {"latency_ms": 260, "ok": True, "tokens": 75},
]


# Step2、使用一致的分母统计成功请求延迟与全部请求错误率
def summarize(records, window_seconds):
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("观察窗口必须为有限正数")
    for record in records:
        if (not math.isfinite(record["latency_ms"]) or record["latency_ms"] < 0
                or type(record["tokens"]) is not int or record["tokens"] < 0
                or type(record["ok"]) is not bool):
            raise ValueError("请求记录的延迟、token 数或成功标记无效")
    success_latencies = sorted(record["latency_ms"] for record in records if record["ok"])
    errors = sum(not record["ok"] for record in records)
    # P95 采用最近秩定义；小样本结果只能用于理解计算过程。
    p95 = success_latencies[math.ceil(0.95 * len(success_latencies)) - 1] if success_latencies else None
    return {"请求数": len(records), "错误率": errors / len(records) if records else None,
            "成功请求平均延迟_ms": mean(success_latencies) if success_latencies else None,
            "成功请求P95_ms": p95, "完成请求数每秒": len(records) / window_seconds,
            "记录的token总数": sum(record["tokens"] for record in records)}


# Step3、展示各项统计，区分服务性能与回答质量
if __name__ == "__main__":
    print(summarize(requests, window_seconds=10))
    # 请求耗时之和不能代替观察窗口：并发请求的耗时可能重叠。
    # 这些指标不衡量答案的准确性，质量评估需使用另外的样本和指标。
