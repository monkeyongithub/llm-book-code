# 文件用途：演示本地PDF混合检索问答
# 对应章节/小节：第8章 8.2.6.1 基于LangChain框架实现RAG
# 作者：NYY
# 创建日期：2024-10-29（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

"""对同一问题比较向量、BM25 与混合检索，再调用本地聊天模型。

本例使用书稿中的 ChatGLM2 或 Qwen-7B-Chat 接口；本地模型文件须提前备好。
"""

import argparse

import jieba
import torch
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["chatglm2", "qwen"], default="chatglm2")
    parser.add_argument("--question", default="保险人是什么意思？")
    args = parser.parse_args()

    # Step1、加载本地 PDF 并分块，保留文档来源元数据
    documents = DirectoryLoader("documents", glob="*.pdf", loader_cls=PyPDFLoader).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80,
                                              separators=["\n\n", "\n", "。", "，", " ", ""])
    chunks = [chunk for chunk in splitter.split_documents(documents) if chunk.page_content.strip()]
    if not chunks:
        raise ValueError("documents 目录中没有可提取的 PDF 文本；扫描件需先 OCR")

    # Step2、建立三种检索方案，比较同一问题命中的文本
    embeddings = HuggingFaceEmbeddings(model_name="m3e-base", encode_kwargs={"normalize_embeddings": True})
    cached = CacheBackedEmbeddings.from_bytes_store(embeddings, LocalFileStore("cache/embeddings"),
                                                    namespace="m3e-base-normalized")
    store = FAISS.from_documents(chunks, cached)
    # T1、向量检索：FAISS 默认按 L2 距离；单位向量下排序等价于余弦相似度排序。
    vector_retriever = store.as_retriever(search_kwargs={"k": 2})
    # T2、BM25 检索：中文文本和查询均使用相同分词方法。
    keyword_retriever = BM25Retriever.from_documents(chunks, preprocess_func=jieba.lcut)
    keyword_retriever.k = 2
    # T3、混合检索：按排名融合，不直接相加两类不同量纲的原始分数。
    hybrid_retriever = EnsembleRetriever(retrievers=[keyword_retriever, vector_retriever], weights=[0.25, 0.75])
    for label, retriever in [("T1 向量", vector_retriever), ("T2 BM25", keyword_retriever), ("T3 混合", hybrid_retriever)]:
        results = retriever.invoke(args.question)
        print(label, [doc.page_content for doc in results])
    retrieved = hybrid_retriever.invoke(args.question)

    # Step3、只加载选定的本地模型，使用其原生 chat 方法
    if not torch.cuda.is_available():
        raise RuntimeError("书稿中的本地 Int4 模型示例需要可用的 CUDA GPU")
    if args.model == "qwen":
        path = "NLP_models/Qwen-7B-Chat-Int4"
        model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True).eval()
    else:
        path = "NLP_models/chatglm2-6b-int4"
        model = AutoModel.from_pretrained(path, trust_remote_code=True).half().cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    context = "\n\n".join(doc.page_content for doc in retrieved)
    prompt = f"请仅根据以下资料回答；资料不足时说明不知道。\n资料：\n{context}\n问题：{args.question}"
    with torch.inference_mode():
        answer, _ = model.chat(tokenizer, prompt, history=[])

    # Step4、显示回答及检索来源，便于读者核对依据
    print(answer)
    for doc in retrieved:
        print(doc.metadata)


if __name__ == "__main__":
    main()
