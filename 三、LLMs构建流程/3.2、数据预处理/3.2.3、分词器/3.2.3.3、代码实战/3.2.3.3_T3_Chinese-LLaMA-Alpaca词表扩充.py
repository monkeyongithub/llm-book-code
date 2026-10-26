# 文件用途：演示Chinese-LLaMA-Alpaca词表扩充
# 对应章节/小节：第3章 3.2.3.3 代码实战
# 作者：NYY
# 创建日期：2024-10-25（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# LLMs之Chinese-LLaMA-Alpaca：源代码解读之merge_tokenizers.py文件实现词表扩充—合并原始的LLaMA分词模型+重训练的中文分词模型(本文直接采用哈工大训练好的中文LLaMA版本)→通过for循环去重把中文语料token扩充到原始LLaMA分词模型中→保存扩充后的分词模型→最后对比两个模型各自对中英文文本的分词效果

'''
https://yunyaniu.blog.csdn.net/article/details/131137574
>> 在通用中文语料上训练了基于sentencepiece的20K中文词表(直接采用哈工大训练好的)并与原版LLaMA模型的32K词表进行合并；通过token去重后，得到的最终LLaMA词表大小为49953；
>> 需要注意的是，在fine-tune阶段Alpaca比LLaMA多一个pad token，所以中文Alpaca的词表大小为49954；
'''

import os

# 指定使用Python实现的Protocol Buffers (一种序列化结构数据的方法)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse

# 0、配置参数
parser = argparse.ArgumentParser()
parser.add_argument('--llama_tokenizer_dir', default=None, type=str, required=True)
parser.add_argument('--chinese_sp_model_file', default='./chinese_sp.model', type=str)
args = parser.parse_args()

# Step1、加载 LLaMA 与中文分词模型
# 从预训练模型中加载LLaMA分词模型
llama_tokenizer_dir = args.llama_tokenizer_dir
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
# 利用SentencePiece加载中文分词模型，即哈工大训练好的开源的中文版LLaMA模型
chinese_sp_model_file = args.chinese_sp_model_file
chinese_sp_model = spm.SentencePieceProcessor()
chinese_sp_model.Load(chinese_sp_model_file)

# Step2、解析两个 SentencePiece 模型并检查特殊标记
# 若使用预训练好的模型，必须进行反序列化
llama_spm = sp_pb2_model.ModelProto()
# 反序列化原始的LLaMA模型：从Llama分词器中获取序列化的SentencePiece模型，并将其反序列化到llama_spm
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
chinese_spm = sp_pb2_model.ModelProto()
# 反序列化中文版LLaMA分词模型：从中文SentencePiece处理器中获取序列化的SentencePiece模型，并将其反序列化到chinese_spm对象中
chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

# 输出词汇表大小、Llama分词器中所有特殊标记、特殊标记对应的ID、特殊标记与其ID之间的映射关系
print(len(llama_tokenizer), len(chinese_sp_model))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)
# 32000、20000
# Llama分词器中所有特殊标记(例如[CLS]/[SEP])：[’<s>‘、’</s>‘、’<unk>‘]，
# 特殊标记对应的ID：[1，2，0]
# 特殊标记与其ID之间的映射关系：{'bos_token开始标记': '<s>',
#                          'eos_token结束标记': '</s>',
#                          'unk_token未知token的标记': '<unk>']

# 将中文 token 添加到 LLaMA 分词器词表
# Step3、去重后扩充 LLaMA 词表
# 统计LLaMA的词汇表，即不同的token共计32000个
llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before:{len(llama_spm_tokens_set)}")

# for循环遍历中文分词器模型(源自哈工大)并去重实现词表扩充：for循环遍历chinese_spm的token并对比是否已存在LLaMA的词汇表中，如果不存在则添加进去
for p in chinese_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        # score 是分词器的片段分数，不是语言模型权重；新增词元仍需扩展嵌入层并训练。
        new_p.score = 0
        llama_spm.pieces.append(new_p)
        llama_spm_tokens_set.add(piece)
print(f"New model pieces: {len(llama_spm.pieces)}")  # 扩充到49953个词汇

llama_spm.trainer_spec.vocab_size = len(llama_spm.pieces)

# Step4、保存 SentencePiece 和 Transformers 格式
output_sp_dir = 'merged_tokenizer_sp'
output_hf_dir = 'merged_tokenizer_hf'  # 设置 Chinese-LLaMA 分词器的保存路径

# SentencePiece 格式：将扩充后的分词模型保存为sentencepiece的二进制文件
# 将合并后的中英文分词器的SentencePiece模型序列化为二进制字符串，并将其写入文件
os.makedirs(output_sp_dir, exist_ok=True)
with open(os.path.join(output_sp_dir, 'chinese_llama.model'), 'wb') as f:
    f.write(llama_spm.SerializeToString())

# Transformers 格式：进一步将分词模型保存为HF格式文件
# 加载模型文件：从刚刚保存的SentencePiece模型文件创建一个新的LlamaTokenizer实例。
tokenizer = LlamaTokenizer(vocab_file=os.path.join(output_sp_dir, 'chinese_llama.model'))
# 将合并后的中英文分词器的Hugging Face格式的模型保存到output_hf_dir目录
tokenizer.save_pretrained(output_hf_dir)
print(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")

# Step5、对比扩充前后的中文分词效果
# 定义测试文本
# 测试分词模型：对比两个模型各自对中文的分词效果
# 原始分词器：测试原始的LLaMA分词模型
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
# 扩充分词器：测试原扩充后的分词模型：对中文更友好，效果更好！
chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)

text = '''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
The primary use of LLaMA is research on large language models,including
'''
print("Test text:\n", text)
print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
print(f"Tokenized by Chinese-LLaMA tokenizer:{chinese_llama_tokenizer.tokenize(text)}")

# 运行脚本
# python scripts/merge_tokenizers.py \
# --llama_tokenizer_dir models/7B_hf \
# --chinese_sp_model_file scripts/chinese_sp.model
