import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from pyitcast1.pyitcast.transformer_utils import Batch
from pyitcast1.pyitcast.transformer_utils import LabelSmoothing
from pyitcast1.pyitcast.transformer_utils import SimpleLossCompute
from pyitcast1.pyitcast.transformer_utils1 import run_epoch
from pyitcast1.pyitcast.transformer_utils import greedy_decode
from pyitcast1.pyitcast.transformer_utils1 import get_std_opt1



# 文本嵌入层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# 定义位置编码器，即也是一个层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-(math.log(10000.0) / d_model)))

        # 这意味着每个位置的频率随着位置的增加而减小。这使得模型能够学习序列中每个位置的重要性。

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# 构建掩码张量
def subsequent_maxk(size):
    attn_shape = (1, size, size)
    subsequent_maxk = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(1 - subsequent_maxk)


# # 2.3.2注意力机制

# 为下面函数重写了注意力机制，否则代码会报错
# 注意力机制代码实现
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask = torch.zeros(1, 1, 1, 1).cuda()
        # print("mask.shape:", mask.shape)
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


# # 2.3.3多头注意力机制

# 实现克隆函数
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 实现多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % head == 0
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
#        print(query.shape)
        if mask is not None:
            mask = mask.unsqueeze(0)
        batch_size = query.size(0)

        # 三个张量分别是三个输入，分别用三个线性层进行处理并重塑维度
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
             for model, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        x = (x.transpose(1, 2).contiguous()
             .view(batch_size, -1, self.head * self.d_k))
        # 拷贝的四个层还有一个就是这个对输入进行线性变换得到输出
        return self.linears[-1](x)


# # 2.3.4前馈全连接层

# 构建前馈全连接网络类
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.dropout(F.relu(self.w1(x)))))


# # 2.3.5规范化层

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


# # 2.3.6子层连接结构

# 构建子层连接结构
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)
        self.size = size

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# # 2.3.7编码器层

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size

        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# # 2.3.8编码器

# 构建编码器类
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# # 2.4解码器

# # 2.4.1解码器层

# 构建解码器层类
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()

        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout

        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))

        return self.sublayer[2](x, self.feed_forward)


# # 2.4.2 解码器

# 构建解码器类
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


# # 2.5输出部分实现
'''
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.shape)
'''

# 构建Generator类
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, d_model, vocal_size):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocal_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=1)


# # 2.6 Transformer模型构建

# 实现编码解码结构
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        return self.decode(self.encode(source, source_mask), source_mask,
                           target, target_mask)

    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.tgt_embed(target), memory, source_mask,
                            target_mask)


# Transformer模型构建过程的代码分析
def make_model(source_vocab, target_vocab, N=6, d_model=768, d_ff=2048, head=16,
               dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(head, d_model)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


# # 2.7模型基本测试运行


# 构建数据集生成器
vocab = 'vocab.txt'
data = 'sample.txt'
max_len = 256

def break_sentence(sentence, max_sent_len):
  """
  For example, for a sentence with 70 words, supposing the the `max_sent_len'
  is 30, break it into 3 sentences.

  :param sentence: list[str] the sentence
  :param max_sent_len:
  :return:
  """
  ret = []
  cur = 0
  length = len(sentence)
  while cur < length:
    if cur + max_sent_len + 5 >= length:
      ret.append(sentence[cur: length])
      break
    ret.append(sentence[cur: min(length, cur + max_sent_len)])
    cur += max_sent_len
  return ret


dic_num_embed = {}
t = 1
f = open(vocab).readlines()
for line in f:
    token = line.split('\n')[0]
    if token not in dic_num_embed.keys():
        dic_num_embed[token] = t
        t += 1
dic_num_embed['*'] = 0
length = []
data = open(data).readlines()
# for line in data:
#     tokens = line.rstrip('\n').split()
#     length.append(len(tokens))
# print(length)
# max_length = max(length)
max_length = int(max_len)
# print(max_length)
data_new = []
for line in data:
    seq = []
    tokens = line.rstrip('\n').split()
    # print(tokens)
    if len(tokens)  > max_length:
        tmp = break_sentence(tokens, max_length)
        for tk_list in tmp:
            seq = []
            for tk in tk_list:
                seq.append(dic_num_embed[tk])
            while len(seq) < max_length:
                seq.append(0)
            if len(seq) > max_length:
                seq = seq[:max_length]
            data_new.append(seq)
    else:
        for i in tokens:
            seq.append(dic_num_embed[i])
        while len(seq) < max_length:
            seq.append(0)
        # if len(seq) > max_length:
        #     seq = seq[:max_length]
        data_new.append(seq)
#  data_new = break_sentence(data_new, max_length)
data_new = torch.LongTensor(data_new)
print('Input data size', data_new.size())
vocab_size = len(list(dic_num_embed.keys()))
print('vocab_size', vocab_size)


V = 100314
batch_size = 16
#num_batch = 30
lr = 2e-4

num_batch = math.ceil(data_new.shape[0]/batch_size)
print('num_batch=',num_batch)
print(type(num_batch))

def data_generator(V, batch_size, num_batch):
    for i in range(num_batch):
        #data = torch.from_numpy(
            #np.random.randint(1, V, size=(batch_size, 10)))
        # data[:, 0] = 1
        # print(data)
     
        # source = Variable(data, requires_grad=False).long()
        # target = Variable(data, requires_grad=False).long()
        
        start = i*batch_size
        end = start+batch_size
        data = data_new[start:end, :]

        source = Variable(data, requires_grad=False).long().cuda()
        target = Variable(data, requires_grad=False).long().cuda()

        yield Batch(source, target)


# 获得Transformer模型及其优化器和损失函数
model = make_model(V, V, N=2)

# 将模型移动到GPU上
model.cuda()

model_optimizer = get_std_opt1(model, lr)

criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
loss = SimpleLossCompute(model.generator, criterion, model_optimizer)


# 运行模型进行训练和评估
def run(model, loss, epochs=50):
    for epoch in range(epochs):
        model.train()
        loss1 = run_epoch(data_generator(V, batch_size, math.ceil(0.8*num_batch)), model, loss)
        print("Epoch: %d Loss: %f" % (epoch, loss1))

        model.eval()
        run_epoch(data_generator(V, batch_size, num_batch-math.ceil(0.8*num_batch)), model, loss)
        torch.save(model, './model_768_2e-4_16/trm_epoch_' + str(epoch) + '.pth')

start = time.time()

run(model, loss)

end = time.time()

# 总时间
total_time = end - start
print(f"Total time: {total_time:.3f}s")

'''
# 使用模型进行贪婪解码
def run(model, loss, epochs=10):
    for epoch in range(epochs):
        model.train()
        run_epoch(data_generator(V, batch_size, 20), model, loss)

        model.eval()
        run_epoch(data_generator(V, batch_size, 5), model, loss)

    model.eval()
    source = torch.LongTensor([[1, 3, 2, 5, 4, 6, 7, 8, 9, 10]]).cuda()

    source_mask = torch.ones(1, 1, 10).cuda()

    result = greedy_decode(model, source, source_mask, max_len=10,
                           start_symbol=1)
    print(result)


start = time.time()

run(model, loss)

end = time.time()

# 总时间
total_time = end - start
print(f"Total time: {total_time:.3f}s")
'''

'''
————————————————

版权声明：本文为博主原创文章，遵循
CC
4.0
BY - SA
版权协议，转载请附上原文出处链接和本声明。

原文链接：https: // blog.csdn.net / qq_42769818 / article / details / 134495605
'''
