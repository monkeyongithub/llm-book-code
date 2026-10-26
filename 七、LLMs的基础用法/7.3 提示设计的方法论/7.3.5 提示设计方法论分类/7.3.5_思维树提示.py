# 文件用途：演示思维树
# 对应章节/小节：第7章 7.3.5 提示设计方法论分类
# 作者：NYY
# 创建日期：2024-03-29（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：使用算术规则模拟思维树搜索流程

"""用确定性的算术搜索模拟思维树的“扩展—评价—保留”流程。

从 1 开始，每步可加 1 或乘 2，尝试在有限深度内到达目标。
候选由规则生成，不调用语言模型；距离评分和束搜索也不保证找到全局最优路径。
"""


def tree_of_thought(target=10, max_depth=5, beam_width=3):
    if target < 1 or max_depth < 0 or beam_width < 1:
        raise ValueError("目标至少为 1，深度非负，保留分支数为正")
    branches = [(1, ["1"])]
    for depth in range(max_depth + 1):
        # Step1、检查当前候选是否已达到目标
        for value, path in branches:
            if value == target:
                return path
        if depth == max_depth:
            break

        # Step2、从每个保留分支生成下一步候选
        candidates = []
        for value, path in branches:
            candidates.append((value + 1, path + [f"{value} + 1 = {value + 1}"]))
            candidates.append((value * 2, path + [f"{value} × 2 = {value * 2}"]))

        # Step3、按距目标的差值评分，去除重复状态后保留少量分支
        candidates.sort(key=lambda item: abs(target - item[0]))
        branches, seen = [], set()
        for value, path in candidates:
            if value not in seen:
                branches.append((value, path))
                seen.add(value)
            if len(branches) == beam_width:
                break
        print(f"第 {depth + 1} 层保留：{[value for value, _ in branches]}")
    return None


# Step4、展示搜索路径；失败仅表示本次深度和剪枝策略下未找到解
if __name__ == "__main__":
    path = tree_of_thought()
    print("\n".join(path) if path else "在当前搜索限制内没有找到解")
