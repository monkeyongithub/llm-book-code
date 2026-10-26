# llm-book-code

> Official companion code repository for *The Dance of Language: A Complete Practical Guide to Large Language Model Applications*

[中文](README.md) | [English](README_EN.md)

---

## 📖 About the Book

- **Title:** *The Dance of Language: A Complete Practical Guide to Large Language Model Applications* (《语言之舞：大语言模型应用实战全书》)
- **Author:** 一个处女座的程序猿 (A Virgo Programmer)
- **Publication:** Mid-2025
- **Writing:** Three years of work; a 680,000-character original manuscript and an approximately 380,000-character final manuscript

## ⭐︎ Five Features of the Book

1. **Theory and practice together**
   - Concise Python examples accompany several chapters, helping readers progress from fundamentals to advanced applications.
2. **Comprehensive, systematic coverage**
   - The book moves from the history and core elements of LLMs through construction, deployment, and monitoring to future trends.
3. **Examples across multiple scenarios**
   - It covers fine-tuning, inference deployment, or online API calls involving GPT-4o, LLaMA-3, ChatGLM-3, GLM-4, and Qwen-2 model families.
4. **A broad view of the toolchain**
   - It covers topics involving Transformers, LangChain, LlamaIndex, Docker, and distributed training.
5. **Advanced augmentation techniques**
   - It covers techniques including RAG, ETA, Agent, and PEFT.

---

## Companion Code at a Glance

This project accompanies 《语言之舞：大语言模型应用实战全书》 (*The Dance of Language: A Complete Practical Guide to Large Language Model Applications*). It helps readers connect the book's concepts, algorithms, and application workflows to concrete code: how inputs are processed, why the key steps are arranged as they are, and what the outputs demonstrate.

The project follows the book's chapter structure and contains **87 Python files, 1 HTML file, and 11 chapter overviews**. Examples cover small algorithm experiments, model calls, fine-tuning and deployment, prompt design, RAG, tool calling, and agents. Some files extract the core workflow of an external project or explain a method through rules and simulated data. Each file's header and its chapter's `CHAPTER_OVERVIEW.md` describe its learning objective and scope.

## Reading the Code Alongside the Book

1. **Read the book section first.** Identify the problem it addresses, the inputs and outputs, and the key concepts.
2. **Open the chapter overview.** Use the section–file–purpose table to locate the example and check the suggested reading order.
3. **Follow the Step comments in the code.** Match each comment to the implementation immediately below it, observing how data and state change.
4. **Compare alternative methods within the section.** When you see `T1`, `T2`, or `T3`, understand the differences before choosing how to run them according to the script's instructions.
5. **Try a small experiment.** Change one input or parameter, predict the result, then observe the output and return to the book to explain it. When evaluating model performance, use consistent data and evaluation criteria.

You can work through the book chapter by chapter or go directly to the section you are reading. Follow the chapter overview for execution order; for example, the FLAN-T5 alignment example requires the adapter produced by the instruction fine-tuning example in the same section.

## Chapter Guide

The table follows the order of the book. Click a chapter title to open its overview and find individual sections and file paths.

| Book chapter | Companion-code focus | Code files |
| --- | --- | --- |
| [Chapter 1: The History of LLMs](<一、大语言模型的发展史/CHAPTER_OVERVIEW.md>) | Bigram language models, text similarity, and model-family calls | 8 Python |
| [Chapter 2: Core Elements of LLM Technology (Data, Architecture, Evaluation, Optimization)](<二、LLMs的核心要素/CHAPTER_OVERVIEW.md>) | Attention, evaluation metrics, and decoding strategies | 5 Python |
| [Chapter 3: The LLM Construction Process (Pretraining → Fine-Tuning → Inference)](<三、LLMs构建流程/CHAPTER_OVERVIEW.md>) | Deduplication, tokenization, input construction, evaluation, fine-tuning, LoRA, and quantization | 13 Python |
| [Chapter 4: LLM Deployment and Monitoring](<四、LLMs的模型部署与监控/CHAPTER_OVERVIEW.md>) | Simulated monitoring metrics, single-machine and DDP training workflows | 2 Python |
| [Chapter 5: Practical LLM Fine-Tuning and Inference Deployment](<五、LLMs的微调与推理部署实战案例/CHAPTER_OVERVIEW.md>) | LLaMA-3, ChatGLM-3, GPT-4o, GLM-4, and Qwen examples | 28 Python + 1 HTML |
| [Chapter 6: Building and Applying LLM Projects](<六、LLMs的构建与应用/CHAPTER_OVERVIEW.md>) | Project lifecycles, enterprise workflows, and application scenarios | Chapter guide |
| [Chapter 7: Basic LLM Usage](<七、LLMs的基础用法/CHAPTER_OVERVIEW.md>) | ICL, CoT, prompt templates, and prompt-method demonstrations | 10 Python |
| [Chapter 8: Advanced and Augmented LLM Usage (RAG, ETA, Agent)](<八、LLMs的进阶与增强用法/CHAPTER_OVERVIEW.md>) | RAG, external tools, agents, and adapter merging | 21 Python |
| [Chapter 9: LLM Training/Inference Frameworks, Deployment Tools, and Prompt Libraries](<九、LLMs的训练推理框架、部署工具和提示库/CHAPTER_OVERVIEW.md>) | Development and training frameworks, deployment tools, and vector databases | Chapter guide |
| [Chapter 10: Challenges and Future Directions](<十、挑战与未来方向/CHAPTER_OVERVIEW.md>) | Bias, fairness, interpretability, resources, and application challenges | Chapter guide |
| [Appendix A: Resources and Tools](<十一、附录：资源和工具/CHAPTER_OVERVIEW.md>) | Learning resources, open-source projects, and tools | Chapter guide |

Chapters 6, 9, and 10 and Appendix A focus on concepts, workflows, or resources. They provide chapter overviews and have no separate code files. Implementations in other chapters can help explain the book's theoretical material.

## File and Comment Conventions

- **Section numbers:** The number at the beginning of a filename identifies the corresponding book section. Refer to the book, file header, and chapter overview for the specific scope.
- **Step1, Step2, Step3:** These mark the example's core processing steps, with comments placed beside the relevant code.
- **T1, T2, T3:** These identify different methods or approaches for the same topic. Some scripts select an approach with `--method`; others demonstrate them in sequence. Read the implementation first.
- **Language and terminology:** Code comments and chapter overviews are primarily in Chinese. Terms such as token, embedding, LoRA, and RAG, along with API fields, function names, and commands, retain their original spelling. The English README uses the same file paths.
- **Resource paths:** Local resource paths in the companion scripts are relative to the project root; run scripts from that directory. Execute external-project commands in the directories specified by their instructions.

Use the following categories to decide how to work with a file:

| Example type | How to read and use it |
| --- | --- |
| Small concept or algorithm example | Focus on inputs, computation, and outputs; install any required libraries before running it. |
| Model, API, training, or service example | Prepare the relevant models, data, dependencies, and service configuration before starting the example. |
| External-project excerpt or procedural guide | Read it alongside the complete project referenced by the book; execute commented commands in the appropriate environment. |
| Rule-based or simulated-data example | Observe the method's structure and data flow; evaluate model effectiveness separately. |
| HTML page template | Use it with the corresponding GLM-4 Flask service in Chapter 5, which loads the page. |

## Start with Two Small Examples

Prepare a Python 3 environment and open a terminal in the project root, the directory containing this README. The following two examples use only the Python standard library. They require no model downloads, API keys, or third-party dependencies.

### Observe Bigram Probabilities and Smoothing

Corresponding book section: **1.1.1 SLM (Statistical Language Models)**.

```bash
python3 "一、大语言模型的发展史/1.1 NLP技术发展的四波浪潮/1.1.1 SLM(统计语言模型)/1.1.1_二元语言模型与平滑.py"
```

Compare next-word distributions under maximum likelihood estimation and add-one smoothing, and observe the probabilities of unseen bigrams and unknown words. Adjust the corpus or smoothing coefficient to understand what it means to reserve probability for unseen events.

### Distinguish Classification Metrics from Perplexity

Corresponding book section: **2.3.1.2 Comparing Common General-Purpose Metrics: ACC, P, R, F1, and PPL**.

```bash
python3 "二、LLMs的核心要素/2.3 模型评估体系—指标与基准/2.3.1 模型评估指标/2.3.1.2 常见的通用指标对比/2.3.1.2_分类指标与困惑度.py"
```

Observe how true and predicted labels determine ACC, P, R, and F1, and how the conditional probabilities of actual target tokens determine PPL. Change predicted labels or target probabilities separately to understand what each metric measures.

If your environment launches Python 3 with `python` or Windows' `py -3`, substitute that command for `python3`. Paths contain spaces, parentheses, and Chinese characters; keep the quotation marks when copying commands.

## Prepare Environments and Resources for Other Examples

### Choose Dependencies for Each Example

Use separate virtual environments for different examples, selecting compatible Python, library, and CUDA versions according to the relevant book example and referenced project. This directory does not provide a single dependency lock file covering all examples.

| Example area | Common components; select those needed by the script |
| --- | --- |
| Text representations and basic computation | NumPy, scikit-learn, NLTK, Gensim, TensorFlow, and Sentence Transformers |
| Tokenization, model inference, and fine-tuning | SentencePiece, PyTorch, Transformers, Datasets, PEFT, TRL, and the quantization or training components required by the example |
| RAG, tools, and agents | LangChain, LangGraph, LlamaIndex, vector stores, document parsers, and the relevant service SDKs |
| Web interfaces and API services | Streamlit, Gradio, FastAPI, Flask, Uvicorn, and LangServe, as required by the example |

This table identifies dependency categories. Select packages according to the chosen script's imports and the referenced project's instructions. Framework APIs depend on versions, especially TRL, LangChain, and custom model interfaces; check that the interfaces used in the book match your local versions.

### Models, Data, and External Projects

- Prepare models, tokenizers, adapters, and data at the paths used by the script. The repository includes some teaching corpora and sample data; obtain complete model weights, external knowledge bases, and third-party projects separately.
- Scripts that support `MODEL_PATH`, `TOKENIZER_PATH`, or command-line path arguments can use those settings to locate resources. For other scripts, follow the paths specified in the file.
- When using excerpts from projects such as LLaMA-Factory, minimind, or ChatGLM, prepare the corresponding project and configuration using the sources referenced in the book. A `.py` file containing only procedural instructions may produce no output when run directly; read its steps and execute the commands as directed.
- Resource requirements for local LLM inference, fine-tuning, and distributed training depend on model size, precision, sequence length, and training approach. Check the example's GPU memory and device requirements before running it.

### API and Service Configuration

Online examples read environment variables for the selected service, such as `OPENAI_API_KEY`, `ZHIPUAI_API_KEY`, `DASHSCOPE_API_KEY`, and `NVIDIA_API_KEY`. Search and graph-database examples also read their own credentials, addresses, or model settings; see the script's `os.environ` / `os.getenv` calls for the exact names.

Configure the variables required by your chosen example and keep credentials in your local environment. Model downloads, web retrieval, and online API calls require network access; service providers may charge based on usage.

## Interpreting Example Results

- **Relate the demonstration's scope to the book's objective.** Single-head attention, small-matrix LoRA, single-step decoding, and simulated monitoring each illustrate part of a mechanism. Follow the book's workflow when extending the implementation.
- **Distinguish real calls from rule-based simulations.** Chapter 7 includes actual model calls as well as prompt templates, candidate search, rule-based revision, and voting examples. A successful run shows that the corresponding workflow executed; effectiveness still needs evaluation on a defined task and dataset.
- **Keep comparison conditions consistent.** Small corpora, a few training steps, or model-assisted scores help explain workflows. When comparing model capabilities, keep datasets, tokenization, metric scales, generation parameters, and other relevant conditions consistent.
- **Understand each example's boundaries.** Chapter 4's DDP example demonstrates data parallelism; the full pipeline deployment discussed in the book requires additional implementation. Chapter 8's adapter-merging example demonstrates weight merging; context capabilities depend on model configuration and adapter training.
- **Use actual resources and results as evidence.** Output comments in the files help explain the code; model sampling, external services, and data contents can affect actual results.

## Sources and Feedback

The book is the reference for chapter content and learning order. Examples involving third-party implementations retain source references in their scripts so readers can explore the complete implementations. Third-party models, data, and projects are subject to their respective licenses.

When reporting a problem, include the book section, relative code path, Python and key dependency versions, minimal reproduction steps, and expected and actual results. These details help locate the issue in the context of the example's learning objective.
