# 文件用途：演示文本分块
# 对应章节/小节：第8章 8.2.3.3 RAG微观+策略角度优化
# 作者：NYY
# 创建日期：2024-10-28（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

from pathlib import Path

from langchain.text_splitter import CharacterTextSplitter


# Step1、准备演示文本或读取本地文本
text = "白日依山尽，黄河入海流。欲穷千里目，更上一层楼"
file_path = Path("data_txt_doc/小学语文—孔乙己.txt")
if file_path.is_file():
    text = file_path.read_text(encoding="utf-8")

# 固定大小分块，并允许相邻块重复一个字符。
# Step2、配置固定大小与重叠量并切分
text_splitter = CharacterTextSplitter(separator="，", chunk_size=10, chunk_overlap=1)
documents = text_splitter.create_documents([text])
# Step3、查看每个文本块
for document in documents:
    print(document.page_content)
