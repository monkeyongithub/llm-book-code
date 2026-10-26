# 文件用途：演示单机训练与多进程分布式训练的基本流程
# 对应章节/小节：第4章 4.3 实战案例
# 作者：NYY
# 创建日期：2024-10-09（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""单机与 DDP 数据并行训练的最小示例。

书稿 4.3 还讨论流水线并行；本例仅演示数据并行，不实现 PP=12 的模型切分。
使用 torchrun 启动多个进程时，每个进程对应一个设备。
"""

import os

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


# Step1、定义简单分类模型
class SimpleModel(nn.Module):
    """将 10 维输入映射到两个类别。"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, features):
        return self.fc(features)


# Step2、配置单机或分布式数据加载
def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if distributed else 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if distributed:
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

    try:
        torch.manual_seed(0)
        features = torch.randn(64, 10)
        labels = torch.randint(0, 2, (64,))
        dataset = TensorDataset(features, labels)
        sampler = DistributedSampler(dataset) if distributed else None
        loader = DataLoader(dataset, batch_size=16, sampler=sampler, shuffle=sampler is None)

        model = SimpleModel().to(device)
        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
            )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Step3、执行训练循环并输出完成状态
        for epoch in range(10):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_features), batch_labels)
                loss.backward()
                optimizer.step()

        if not distributed or torch.distributed.get_rank() == 0:
            print("训练完成。")
    finally:
        if distributed:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
