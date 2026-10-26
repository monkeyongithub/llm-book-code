# 文件用途：说明lm-evaluation-harness的C-Eval评测
# 对应章节/小节：第3章 3.3.3.2 代码实战：基准评估案例（C-Evals）
# 作者：NYY
# 创建日期：2024-10-27（文件系统记录）
# 运行方式：阅读示例说明
# 备注：操作说明，不包含独立执行逻辑

# Step1、准备评测框架、C-Eval 数据和本地模型
# 本文件是命令说明。框架与模型准备完毕后，从配书代码根目录运行下列命令。
# 评测框架来源：https://github.com/EleutherAI/lm-evaluation-harness

# Step2、选择 C-Eval 验证任务并保存评测结果
# lm_eval --model vllm \
#     --model_args pretrained=models/glm-4-9b-chat \
#     --tasks ceval-valid \
#     --batch_size 1 \
#     --output_path output/ceval_result \
#     --log_samples

# Step3、阅读任务分数与逐题结果
# 分数受模型、提示模板和评测设置影响，以实际输出为准。
# 使用相同的数据划分和设置比较模型，避免把不同条件下的分数直接相减。
