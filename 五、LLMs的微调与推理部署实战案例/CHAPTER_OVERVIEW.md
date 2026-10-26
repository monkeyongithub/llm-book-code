# 第5章 LLM的微调与推理部署实战案例

本章围绕 LLaMA-3、ChatGLM-3、GPT-4o、GLM-4 和 Qwen 的微调、推理部署及在线调用展开。按具体模型与使用方式选择示例。

<!-- CODE_OVERVIEW:START -->
## 子章节与配套代码

| 书稿小节 | 代码文件 | 用途 |
| --- | --- | --- |
| 5.1.1.1 官方Demo案例——基于指令数据集利用LoRA微调Llama-3-8B-Instruct并生成推理对话 | [`5.1、基于LLaMA-3模型实战/5.1.1、微调实战/5.1.1.1 官方Demo案例——基于指令数据集利用LoRA微调Llama-3-8B-Instruct并生成推理对话/5.1.1.1_LLaMA3官方LoRA微调.py`](<5.1、基于LLaMA-3模型实战/5.1.1、微调实战/5.1.1.1 官方Demo案例——基于指令数据集利用LoRA微调Llama-3-8B-Instruct并生成推理对话/5.1.1.1_LLaMA3官方LoRA微调.py>) | 演示LLaMA-3指令微调 |
| 5.1.1.2 基于Colab平台采用LLaMA-Factory工具和unsloth优化框架实现LoRA微调llama-3-8b-Instruct | [`5.1、基于LLaMA-3模型实战/5.1.1、微调实战/5.1.1.2 基于Colab平台采用LLaMA-Factory工具和unsloth优化框架实现LoRA微调llama-3-8b-Instruct/5.1.1.2_Colab版LLaMA-Factory微调.py`](<5.1、基于LLaMA-3模型实战/5.1.1、微调实战/5.1.1.2 基于Colab平台采用LLaMA-Factory工具和unsloth优化框架实现LoRA微调llama-3-8b-Instruct/5.1.1.2_Colab版LLaMA-Factory微调.py>) | 分阶段生成训练配置、测试已训练适配器与生成导出配置 |
| 5.1.1.3 采用LLaMA-Factory工具GUI的方式实现微调Llama-3-8B-Instruct：采用ShareGPT格式中文对话微调数据集 | [`5.1、基于LLaMA-3模型实战/5.1.1、微调实战/5.1.1.3 采用LLaMA-Factory工具GUI的方式实现微调Llama-3-8B-Instruct：采用ShareGPT格式中文对话微调数据集/5.1.1.3_LLaMA-Factory图形界面微调说明.py`](<5.1、基于LLaMA-3模型实战/5.1.1、微调实战/5.1.1.3 采用LLaMA-Factory工具GUI的方式实现微调Llama-3-8B-Instruct：采用ShareGPT格式中文对话微调数据集/5.1.1.3_LLaMA-Factory图形界面微调说明.py>) | 说明LLaMA-Factory图形界面微调 |
| 5.1.1.4 基于Colab平台利用unsloth优化框架实现LoRA微调llama-3-8b | [`5.1、基于LLaMA-3模型实战/5.1.1、微调实战/5.1.1.4 基于Colab平台利用unsloth优化框架实现LoRA微调llama-3-8b/5.1.1.4_Unsloth微调LLaMA3.py`](<5.1、基于LLaMA-3模型实战/5.1.1、微调实战/5.1.1.4 基于Colab平台利用unsloth优化框架实现LoRA微调llama-3-8b/5.1.1.4_Unsloth微调LLaMA3.py>) | 演示unsloth微调 |
| 5.1.2.1 快速使用LLaMA3-8B-Instruct进行推理测试 | [`5.1、基于LLaMA-3模型实战/5.1.2、推理部署实战/5.1.2.1 快速使用LLaMA3-8B-Instruct进行推理测试/5.1.2.1_LLaMA3推理方式对比.py`](<5.1、基于LLaMA-3模型实战/5.1.2、推理部署实战/5.1.2.1 快速使用LLaMA3-8B-Instruct进行推理测试/5.1.2.1_LLaMA3推理方式对比.py>) | 按 --method 选择 T1 直接生成、T2 pipeline 或 T3 在线接口 |
| 5.1.2.2 LLaMA3-8B-Instruct WebDemo 部署 | [`5.1、基于LLaMA-3模型实战/5.1.2、推理部署实战/5.1.2.2 LLaMA3-8B-Instruct WebDemo 部署/5.1.2.2_LLaMA3网页对话.py`](<5.1、基于LLaMA-3模型实战/5.1.2、推理部署实战/5.1.2.2 LLaMA3-8B-Instruct WebDemo 部署/5.1.2.2_LLaMA3网页对话.py>) | 演示LLaMA-3网页对话 |
| 5.1.2.3 采用FastApi实现部署调用Llama-3-8B-Instruct | [`5.1、基于LLaMA-3模型实战/5.1.2、推理部署实战/5.1.2.3 采用FastApi实现部署调用Llama-3-8B-Instruct/5.1.2.3_LLaMA3_FastAPI对话服务.py`](<5.1、基于LLaMA-3模型实战/5.1.2、推理部署实战/5.1.2.3 采用FastApi实现部署调用Llama-3-8B-Instruct/5.1.2.3_LLaMA3_FastAPI对话服务.py>) | 请求校验、模型生命周期、聊天模板与客户端历史传递 |
| 5.1.3.1 采用官方API接口方式调用——T1、基于Nvidia官网API实现 | [`5.1、基于LLaMA-3模型实战/5.1.3、在线测试使用实战/5.1.3.1 采用官方API接口方式调用——T1、基于Nvidia官网API实现/5.1.3.1_NVIDIA接口调用LLaMA3.py`](<5.1、基于LLaMA-3模型实战/5.1.3、在线测试使用实战/5.1.3.1 采用官方API接口方式调用——T1、基于Nvidia官网API实现/5.1.3.1_NVIDIA接口调用LLaMA3.py>) | 演示LLaMA-3在线API调用 |
| 5.2.1.1 官方Demo案例——基于AdvertiseGen数据集利用单卡GPU并采用Ptuning_v2/LoRA进行微调 | [`5.2、基于ChatGLM-3模型实战/5.2.1、微调实战/5.2.1.1 官方Demo案例——基于AdvertiseGen数据集利用单卡GPU并采用Ptuning_v2／LoRA进行微调/5.2.1.1_ChatGLM3_AdvertiseGen微调.py`](<5.2、基于ChatGLM-3模型实战/5.2.1、微调实战/5.2.1.1 官方Demo案例——基于AdvertiseGen数据集利用单卡GPU并采用Ptuning_v2／LoRA进行微调/5.2.1.1_ChatGLM3_AdvertiseGen微调.py>) | 演示AdvertiseGen微调 |
| 5.2.1.2 基于官方Demo微调——多卡全量微调/单卡P-tuning_v2微调 | [`5.2、基于ChatGLM-3模型实战/5.2.1、微调实战/5.2.1.2 基于官方Demo微调——多卡全量微调／单卡P-tuning_v2微调/5.2.1.2_ChatGLM3官方对话微调.py`](<5.2、基于ChatGLM-3模型实战/5.2.1、微调实战/5.2.1.2 基于官方Demo微调——多卡全量微调／单卡P-tuning_v2微调/5.2.1.2_ChatGLM3官方对话微调.py>) | 说明官方对话微调 |
| 5.2.1.3 利用云服务器(4090-24G)采用P-Tuning v2微调 | [`5.2、基于ChatGLM-3模型实战/5.2.1、微调实战/5.2.1.3 利用云服务器(4090-24G)采用P-Tuning v2微调/5.2.1.3_ChatGLM3_P-Tuning微调说明.py`](<5.2、基于ChatGLM-3模型实战/5.2.1、微调实战/5.2.1.3 利用云服务器(4090-24G)采用P-Tuning v2微调/5.2.1.3_ChatGLM3_P-Tuning微调说明.py>) | 说明P-Tuning v2微调 |
| 5.2.1.4 基于LLaMA-Factory框架并采用GUI方式微调 | [`5.2、基于ChatGLM-3模型实战/5.2.1、微调实战/5.2.1.4 基于LLaMA-Factory框架并采用GUI方式微调/5.2.1.4_ChatGLM3_LLaMA-Factory微调说明.py`](<5.2、基于ChatGLM-3模型实战/5.2.1、微调实战/5.2.1.4 基于LLaMA-Factory框架并采用GUI方式微调/5.2.1.4_ChatGLM3_LLaMA-Factory微调说明.py>) | 说明LLaMA-Factory微调 |
| 5.2.2.1 采用官方Demo的CLI的形式实现对话交互 | [`5.2、基于ChatGLM-3模型实战/5.2.2、推理部署实战/5.2.2.1 采用官方Demo的CLI的形式实现对话交互/5.2.2.1_ChatGLM3命令行对话.py`](<5.2、基于ChatGLM-3模型实战/5.2.2、推理部署实战/5.2.2.1 采用官方Demo的CLI的形式实现对话交互/5.2.2.1_ChatGLM3命令行对话.py>) | 演示ChatGLM-3命令行对话 |
| 5.2.2.2 采用官方Demo的Web界面实现对话模式、工具模式、代码解释器模式 | [`5.2、基于ChatGLM-3模型实战/5.2.2、推理部署实战/5.2.2.2 采用官方Demo的Web界面实现对话模式、工具模式、代码解释器模式/5.2.2.2_ChatGLM3网页对话.py`](<5.2、基于ChatGLM-3模型实战/5.2.2、推理部署实战/5.2.2.2 采用官方Demo的Web界面实现对话模式、工具模式、代码解释器模式/5.2.2.2_ChatGLM3网页对话.py>) | 按 --ui 选择 Gradio 或 Streamlit 的基础多轮对话 |
| 5.2.2.3 采用基于LangChain框架实现ChatGLM3的工具调用功能 | [`5.2、基于ChatGLM-3模型实战/5.2.2、推理部署实战/5.2.2.3 采用基于LangChain框架实现ChatGLM3的工具调用功能/5.2.2.3_ChatGLM3_LangChain工具调用.py`](<5.2、基于ChatGLM-3模型实战/5.2.2、推理部署实战/5.2.2.3 采用基于LangChain框架实现ChatGLM3的工具调用功能/5.2.2.3_ChatGLM3_LangChain工具调用.py>) | 演示ChatGLM-3工具调用 |
| 5.2.3.1 采用官方API接口方式调用 | [`5.2、基于ChatGLM-3模型实战/5.2.3、在线测试使用实战/5.2.3.1 采用官方API接口方式调用/5.2.3.1_智谱接口调用ChatGLM3.py`](<5.2、基于ChatGLM-3模型实战/5.2.3、在线测试使用实战/5.2.3.1 采用官方API接口方式调用/5.2.3.1_智谱接口调用ChatGLM3.py>) | 演示ChatGLM-3在线API调用 |
| 5.3.1 基于官方API在线推理测试 | [`5.3、基于GPT-4o模型实战/5.3.1 基于官方API在线推理测试/5.3.1_GPT4o接口调用.py`](<5.3、基于GPT-4o模型实战/5.3.1 基于官方API在线推理测试/5.3.1_GPT4o接口调用.py>) | 演示GPT-4o API调用 |
| 5.4.1.1 基于官方demo案例依次采用lora(单机单卡)、p-tuning v2(单机单卡)、SFT(单机8卡)实现微调 | [`5.4、基于GLM-4模型实战/5.4.1、微调实战/5.4.1.1 基于官方demo案例依次采用lora(单机单卡)、p-tuning v2(单机单卡)、SFT(单机8卡)实现微调/5.4.1.1_GLM4官方微调方案.py`](<5.4、基于GLM-4模型实战/5.4.1、微调实战/5.4.1.1 基于官方demo案例依次采用lora(单机单卡)、p-tuning v2(单机单卡)、SFT(单机8卡)实现微调/5.4.1.1_GLM4官方微调方案.py>) | 说明GLM-4微调流程 |
| 5.4.1.2 利用云服务器(4090-24G)采用LoRA或P-Tuning v2实现对话微调 | [`5.4、基于GLM-4模型实战/5.4.1、微调实战/5.4.1.2 利用云服务器(4090-24G)采用LoRA或P-Tuning v2实现对话微调/5.4.1.2_GLM4云端对话微调.py`](<5.4、基于GLM-4模型实战/5.4.1、微调实战/5.4.1.2 利用云服务器(4090-24G)采用LoRA或P-Tuning v2实现对话微调/5.4.1.2_GLM4云端对话微调.py>) | 说明GLM-4微调流程 |
| 5.4.1.2 利用云服务器(4090-24G)采用LoRA或P-Tuning v2实现对话微调 | [`5.4、基于GLM-4模型实战/5.4.1、微调实战/5.4.1.2 利用云服务器(4090-24G)采用LoRA或P-Tuning v2实现对话微调/5.4.1.2_问答数据转ChatML.py`](<5.4、基于GLM-4模型实战/5.4.1、微调实战/5.4.1.2 利用云服务器(4090-24G)采用LoRA或P-Tuning v2实现对话微调/5.4.1.2_问答数据转ChatML.py>) | 校验并保留多行问答，转换为 ChatML JSONL |
| 5.4.2.1 基于官方Demo实现基础用法——基于transformers或vLLM后端并采用 GLM-4-9B 模型实现对话交互 | [`5.4、基于GLM-4模型实战/5.4.2、推理部署实战/5.4.2.1 基于官方Demo实现基础用法——基于transformers或vLLM后端并采用 GLM-4-9B 模型实现对话交互/5.4.2.1_GLM4推理后端对比.py`](<5.4、基于GLM-4模型实战/5.4.2、推理部署实战/5.4.2.1 基于官方Demo实现基础用法——基于transformers或vLLM后端并采用 GLM-4-9B 模型实现对话交互/5.4.2.1_GLM4推理后端对比.py>) | 按 --method 选择 Transformers 或 vLLM 后端 |
| 5.4.2.2 基于官方Demo实现复杂用法——基于Web的方式构建与GLM-4-9B模型交互并实现All Tools、文档解读等功能 | [`5.4、基于GLM-4模型实战/5.4.2、推理部署实战/5.4.2.2 基于官方Demo实现复杂用法——基于Web的方式构建与GLM-4-9B模型交互并实现All Tools、文档解读等功能/5.4.2.2_GLM4网页工具交互说明.py`](<5.4、基于GLM-4模型实战/5.4.2、推理部署实战/5.4.2.2 基于官方Demo实现复杂用法——基于Web的方式构建与GLM-4-9B模型交互并实现All Tools、文档解读等功能/5.4.2.2_GLM4网页工具交互说明.py>) | 演示GLM-4网页和工具交互 |
| 5.4.2.3 基于云服务器(4090-24G)实现一键部署开启服务并交互测试 | [`5.4、基于GLM-4模型实战/5.4.2、推理部署实战/5.4.2.3 基于云服务器(4090-24G)实现一键部署开启服务并交互测试/5.4.2.3_GLM4_Flask对话服务.py`](<5.4、基于GLM-4模型实战/5.4.2、推理部署实战/5.4.2.3 基于云服务器(4090-24G)实现一键部署开启服务并交互测试/5.4.2.3_GLM4_Flask对话服务.py>) | 加载基础模型或适配器，处理各浏览器独立的聊天历史 |
| 5.4.2.3 基于云服务器(4090-24G)实现一键部署开启服务并交互测试 | [`5.4、基于GLM-4模型实战/5.4.2、推理部署实战/5.4.2.3 基于云服务器(4090-24G)实现一键部署开启服务并交互测试/templates/5.4.2.3_GLM4聊天页面.html`](<5.4、基于GLM-4模型实战/5.4.2、推理部署实战/5.4.2.3 基于云服务器(4090-24G)实现一键部署开启服务并交互测试/templates/5.4.2.3_GLM4聊天页面.html>) | 发送消息与当前页面历史，展示回复并处理请求失败 |
| 5.4.3.1 采用官方API接口方式调用ChatGLM-4-plus | [`5.4、基于GLM-4模型实战/5.4.3、在线测试使用实战/5.4.3.1 采用官方API接口方式调用ChatGLM-4-plus/5.4.3.1_智谱接口调用GLM4.py`](<5.4、基于GLM-4模型实战/5.4.3、在线测试使用实战/5.4.3.1 采用官方API接口方式调用ChatGLM-4-plus/5.4.3.1_智谱接口调用GLM4.py>) | 演示GLM-4在线API调用 |
| 5.5.1 微调实战——基于LLaMA-Factory框架微调Qwen-2模型 | [`5.5、基于Qwen-2模型实战/5.5.1、微调实战/5.5.1_Qwen2_LLaMA-Factory微调说明.py`](<5.5、基于Qwen-2模型实战/5.5.1、微调实战/5.5.1_Qwen2_LLaMA-Factory微调说明.py>) | 说明Qwen-2的LLaMA-Factory微调 |
| 5.5.2.1 基于Transformers框架实现测试Qwen2.5-7B-Instruct模型 | [`5.5、基于Qwen-2模型实战/5.5.2、推理部署实战/5.5.2.1 基于Transformers框架实现测试Qwen2.5-7B-Instruct模型/5.5.2.1_Qwen2_Transformers推理.py`](<5.5、基于Qwen-2模型实战/5.5.2、推理部署实战/5.5.2.1 基于Transformers框架实现测试Qwen2.5-7B-Instruct模型/5.5.2.1_Qwen2_Transformers推理.py>) | 演示Qwen模型推理 |
| 5.5.2.2 基于Ollama来部署Qwen2.5-7B模型 | [`5.5、基于Qwen-2模型实战/5.5.2、推理部署实战/5.5.2.2 基于Ollama来部署Qwen2.5-7B模型/5.5.2.2_Qwen2_Ollama部署.py`](<5.5、基于Qwen-2模型实战/5.5.2、推理部署实战/5.5.2.2 基于Ollama来部署Qwen2.5-7B模型/5.5.2.2_Qwen2_Ollama部署.py>) | 说明Ollama部署Qwen模型 |
| 5.5.3.1 API接口调用 | [`5.5、基于Qwen-2模型实战/5.5.3、在线测试使用实战/5.5.3.1 API接口调用/5.5.3.1_Qwen在线接口调用.py`](<5.5、基于Qwen-2模型实战/5.5.3、在线测试使用实战/5.5.3.1 API接口调用/5.5.3.1_Qwen在线接口调用.py>) | 演示Qwen在线API调用 |

## 建议阅读顺序

先选择模型小节，再按微调、推理部署、在线调用阅读。各模型案例相互独立。多方案文件通过参数选择其一，避免同时启动多个服务或加载多个后端。

## 注意事项

- 资源路径按代码根目录解释；外部项目的命令说明须结合对应项目使用，配书目录不包含其完整源码。
- Colab 的 LLaMA-Factory 示例先选择 `--stage prepare` 生成配置，完成 CLI 训练后选择 `--stage chat`；导出配置使用 `--stage export-config`。
- LLaMA-3 推理比较使用 `--method T1/T2/T3`；GLM-4 后端比较使用 `--method T1/T2`。
- ChatGLM-3 网页示例默认 Gradio；Streamlit 启动格式为 `streamlit run <脚本相对路径> -- --ui streamlit`。工具和代码解释器模式需阅读书稿指定的 composite_demo 项目。
- GLM-4 Flask 服务与同目录 templates 下的 HTML 配套使用；每个浏览器携带自己的历史。FastAPI 示例的客户端应在服务启动后的另一个进程中调用。
- 纯操作说明脚本中的终端命令和数据结构以注释保留，不会因为运行该 Python 文件而自动执行。
<!-- CODE_OVERVIEW:END -->
