# 文件用途：演示BELLE分词器训练与合并
# 对应章节/小节：第3章 3.2.3.3 代码实战
# 作者：NYY
# 创建日期：2024-10-25（文件系统记录）
# 运行方式：按书稿对应小节使用
# 备注：核心演示代码

# LLMs之BELLE：源码解读(merge_tokenizers.py文件)训练和合并两个不同的SentencePiece分词模型—使用SentencePiece库来训练一个名为belle的BPE分词器→加载两个现有的分词器模型→合并词汇表→保存合并后的模型作为新的分词器模型
'''训练并合并分词器词表。'''
import sentencepiece as spm

# Step1、训练 BELLE 分词器
# input: 指定了训练模型所需的输入文本文件的路径。
# model_prefix: 定义了模型文件的前缀名称，将生成两个文件，一个是belle.model用于保存模型配置，另一个是belle.vocab用于保存词汇表。
# model_type: 这里设置为'bpe'，表示使用BPE (Byte Pair Encoding) 分词算法。
# vocab_size: 指定了词汇表的大小，这里设置为25000。
# character_coverage: 指定了覆盖字符的比例，这里设置为0.9995，表示要尽量覆盖99.95%的字符。
spm.SentencePieceTrainer.train(input='三、LLMs构建流程/3.2、数据预处理/3.2.3、分词器/3.2.3.3、代码实战/sentences.txt',
                               model_prefix='belle',
                               model_type='bpe',
                               vocab_size=25000,
                               hard_vocab_limit=False,
                               character_coverage=0.9995)

# Step2、加载原始和 BELLE 分词模型
# 加载待合并的分词模型
from sentencepiece import sentencepiece_model_pb2 as model

orig_model_path = 'models/llama/tokenizer.model'
belle_model_path = 'belle.model'
orig_m = model.ModelProto()
belle_m = model.ModelProto()
# 使用ParseFromString方法从模型文件中读取模型配置信息，将信息解析到这两个对象中
orig_m.ParseFromString(open(orig_model_path, "rb").read())
belle_m.ParseFromString(open(belle_model_path, "rb").read())
print(len(orig_m.pieces), len(belle_m.pieces))

# Step3、合并两个模型的词表
# 首先，遍历了两个模型的词汇表，将orig_model的词汇表片段存入orig_pieces列表
# 然后，遍历belle_model的词汇表片段，并检查belle_m中的词汇片段是否已经存在于orig_pieces中，
# 如果不存在，就将这个词汇片段添加到orig_m的词汇表中，同时也将它添加到orig_pieces列表中。
orig_pieces = []
for piece in orig_m.pieces:
    orig_pieces.append(piece.piece)
for piece in belle_m.pieces:
    if piece.piece not in orig_pieces:
        orig_m.pieces.append(piece)
        orig_pieces.append(piece.piece)
print(len(orig_m.pieces))

# 词表扩充后需要同步扩展语言模型的嵌入层并训练新增词元。
orig_m.trainer_spec.vocab_size = len(orig_m.pieces)

# Step4、保存扩充后的分词模型
save_vocab_path = 'merged_tokenizer.model'
with open(save_vocab_path, 'wb') as f:
    f.write(orig_m.SerializeToString())
