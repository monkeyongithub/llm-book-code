# -*- coding: utf-8 -*-
# 文件用途：演示SentencePiece分词模型训练
# 对应章节/小节：第3章 3.2.3.3 代码实战
# 作者：NYY
# 创建日期：2024-10-25（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

'''

https://yunyaniu.blog.csdn.net/article/details/132867310
'''

# Step1、准备领域语料
text = r"由于ChatGPT、GPT-4近期火爆整个互联网，掀起了人工智能相关的二次开发应用的热潮，博主同时也应广大网友私信请求(太多了，无法一一回复，感谢理解！！)。在本文章中，博主将会及时、持续地更新人工智能领域最新的PTMs算法模型，以及LLMs大模型的部署实战案例。同时，博主也会特持续收集很多基于AI的产品合集，以方便广大网友测试和测试，并同时反馈产品效果，博主会及时更新产品排序。如果大家有新的AI工具，也可留言，博主会将留言的提到的AI小工具，加入到本文章内容。"
sentences = text.split("。")
# 打开一个txt文件，将句子写入文件
with open("sentences.txt", "w", encoding="utf-8") as file:
    for sentence in sentences:
        file.write(sentence + "。\n")
print("文本已拆分并保存到sentences.txt文件中。")


# Step2、训练 SentencePiece 分词器并查看词表
# 使用 SentencePiece 训练分词模型
model_name = 'tokenizer'
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='sentences.txt',
    model_prefix=model_name,
    vocab_size=450,
    hard_vocab_limit=False,  # 小语料允许实际词表小于设定上限。
    user_defined_symbols=['foo', 'bar'],  # 自定义的符号
    character_coverage=1.0,
    model_type="bpe",
)
# 运行后会得到tokenizer.model和tokenizer.vocab两个文件

# 查看生成的词汇表
vocab_path = "%s.vocab"% model_name
with open(vocab_path, "r", encoding="utf-8") as file:
    vocab_contents = file.read()
print(vocab_contents)


# Step3、加载模型并测试分词
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("tokenizer.model")
text = "由于ChatGPT、GPT-4近期火爆整个互联网。"
tokens = sp.EncodeAsPieces(text)
print(tokens)
