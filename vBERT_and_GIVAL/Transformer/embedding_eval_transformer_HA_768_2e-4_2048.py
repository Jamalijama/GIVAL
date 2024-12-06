from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, \
    normalized_mutual_info_score, accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

#from imblearn.over_sampling import SMOTE
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
from pyitcast1.pyitcast.transformer_utils1 import run_epoch1
from pyitcast1.pyitcast.transformer_utils import greedy_decode
from pyitcast1.pyitcast.transformer_utils import get_std_opt



# 构建数据集生成器
vocab = 'vocab.txt'
data = './data/try_AIV_sample_token_onlywith_HMM_CDS_4_HA_serotype2000.txt'
max_len = 257

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
                if tk not in dic_num_embed.keys():
                    seq.append(dic_num_embed['UNK'])
                else:
                    seq.append(dic_num_embed[tk])
            while len(seq) < max_length:
                seq.append(0)
            if len(seq) > max_length:
                seq = seq[:max_length]
            data_new.append(seq)
    else:
        for i in tokens:
            if i not in dic_num_embed.keys():
                seq.append(dic_num_embed['UNK'])
            else:
                seq.append(dic_num_embed[i])
        while len(seq) < max_length:
            seq.append(0)
        # if len(seq) > max_length:
        #     seq = seq[:max_length]
        data_new.append(seq)
#  data_new = break_sentence(data_new, max_length)
#  data_new = torch.LongTensor(data_new)


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

# 构建Generator类
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, d_model, vocal_size):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocal_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=1)

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


V = 11
checkpoint = "./model_768_2e-4_16/trm_epoch_29.pth"
#model = make_model(V, V, N=2)
#model.load_state_dict(torch.load(checkpoint))
model = torch.load(checkpoint)
# print(model)

inputs = data_new
outputs = []
for i, input in enumerate(inputs):

    input = torch.LongTensor([input]).cuda()
    source = Variable(input, requires_grad=False).long().cuda()
    target = Variable(input, requires_grad=False).long().cuda() 
    model.eval()
    # output = model(Batch(source, target))
    output = run_epoch1(Batch(source, target), model)
    if i <= 5:
        print('out')
        print(Batch(source, target).trg)
    outputs.append(output.cpu().detach().numpy())
    
np_outputs = np.array(outputs)
np_outputs = np_outputs.reshape(np_outputs.shape[0], np_outputs.shape[2], -1)
print(np_outputs.shape)



maxlen=256
word_dim=768
gene = 'HA'
method_name = 'HA2000_transformer_768_2e-4_2048'
path1 = '768_2e-4_2048/'

'''
#padding
word_feature_len = 768
max_token_num = 185
len_max = max_token_num * word_feature_len
input_array0 = np.load('try_AIV_sample_token_onlywith_HMM_CDS_4_HA_serotype2000_38w_3time.npy', allow_pickle=True)
method_name = '2000sample_serotype_new_AIV_HA_HMM_256cut_3time'



#name = '2000_serotype_new_sample_AIV_10w_data_30w_epoch_256cut_model_HA'
print(input_array0[0])
# input_array = np.array([[4, 10, 5], [2], [3, 7, 9], [2, 5]])
#input_array = [[4, 1, 5], [2], [3, 7, 9], [2, 5]]
input_array = []
for i in range(len(input_array0)):
#for i in range(10):
    print(i)
    seq_composition = input_array0[i]
    seq_composition_len_reshape = len(seq_composition) * word_feature_len
    reshape_composition0 = np.reshape(seq_composition,(1,seq_composition_len_reshape))
    reshape_composition = list(reshape_composition0[0])
    input_array.append(reshape_composition)
print(len(input_array))
print(len(input_array[0]))
matrix_npy = tf.keras.preprocessing.sequence.pad_sequences(input_array, maxlen=None, dtype='object',padding='post')


#method_name = '2000sample_serotype_new_AIV_HA_10w_data_30w_epoch_256cut_model_minibatch_kmeans_pad_0'
#matrix_npy = np.load('feature_after_pad_seq_2000_serotype_new_sample_AIV_10w_data_38w_epoch_256cut_model_HA.npy', allow_pickle=True)
'''

matrix_npy = np_outputs
df_all_information = pd.read_csv('./embedding_eval/HA/HA_sampled_Serotype_new_not_in_training_set_and_1wHA.csv')
label_1 = list(df_all_information['Serotype_new'])

cnames = {
'lightblue':            '#ADD8E6',
'deepskyblue':          '#00BFFF',
'cadetblue':            '#5F9EA0',
'cyan':                 '#00FFFF',
'purple':               '#800080',
'orchid':               '#DA70D6',
'lightgreen':           '#90EE90',
'darkgreen':            '#006400',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32',
'deeppink':             '#FF1493',
'burlywood':            '#DEB887',
'red':                  '#FF0000',
'indianred':            '#CD5C5C',
'darkred':              '#8B0000',
    }
# cnames = {
# 'darkgreen':            '#006400',
# # 'lightblue':            '#ADD8E6',
# 'yellow':               '#FFFF00',
# 'yellowgreen':          '#9ACD32',
# 'burlywood':            '#DEB887',
# 'blue1':                '#1663A9',
# 'gray':                 '#666666',
# 'darkgreen':            '#006400',
# 'darkyellow':           '#996600',
# 'purple':               '#800080',
# # 'darkred':              '#8B0000',
# 'deeppink':             '#FF1493',
# 'green':                '#66CC00',
# 'deepskyblue':          '#00BFFF',
# 'orange1':              '#FF6600',
# 'orchid':               '#DA70D6',
# 'yellow':               '#FFCC33',
# 'blue2':                '#6EB1DE',
# 'lightgreen':           '#90EE90',
# # 'blue3':                '#1383C2',
# # 'blue4':                '#20C2L1',
# # 'deepskyblue':          '#00BFFF',
#
#     }

color_num_list = list (range(1,16,1))

# print (len(color_num_list))
color_dict = dict(zip(color_num_list,cnames.values()))
# print (color_dict)
color_list0 = list(color_dict.values())

y_types = sorted(set(label_1))
y_num = len(y_types)

# a = np.array([[111,222],[333,444]])
# b = np.reshape(a,(1,4))
# print(b)

reshape_composition_lst = []
Serotype_new_lst = []


for i in range(len(matrix_npy)):
    sample_composition = matrix_npy[i]
    reshape_composition0 = np.reshape(sample_composition,(1,maxlen*word_dim))
    reshape_composition = reshape_composition0[0]
    reshape_composition_lst.append(reshape_composition)
    # print(reshape_composition)
    Serotype_new = df_all_information.loc[i,'Serotype_new']
    Serotype_new_lst.append(Serotype_new)


reshape_composition_array = np.array(reshape_composition_lst)
Serotype_new_array = np.array(Serotype_new_lst)
X_tsne = TSNE(learning_rate=100,random_state=10).fit_transform(reshape_composition_array)
#X_pca = PCA(n_components = 2).fit_transform(reshape_composition_array)
k_selected=len(y_types)
ac_cluster_1_pred = MiniBatchKMeans(n_clusters=k_selected, random_state=10).fit_predict(X_tsne)

#ac_cluster_1_pred = AgglomerativeClustering(n_clusters=2, linkage='complete').fit_predict(X_pca)
# # 聚类效果评价
# print('\nagglomerative clustering:')
# # accuracy，值越大越好
# acc_1 = accuracy_score(label_1, ac_cluster_1_pred)
# print('accuracy=',acc_1)
# # 轮廓系数，值越大越好
sh_score_1 = silhouette_score(X_tsne, ac_cluster_1_pred, metric='euclidean')
print('sh_score_1=',sh_score_1)
# # Calinski-Harabaz Index，值越大越好
ch_score_1 = calinski_harabasz_score(X_tsne, ac_cluster_1_pred)
print('ch_score_1=',ch_score_1)
# #  Davies-Bouldin Index(戴维森堡丁指数)，值越小越好
dbi_score_1 = davies_bouldin_score(X_tsne, ac_cluster_1_pred)
print('dbi_score_1=',dbi_score_1)
# # 调整兰德指数，值越大越好
ari_score_1 = adjusted_rand_score(label_1, ac_cluster_1_pred)
print('ari_score_1=',ari_score_1)
# # 标准化互信息，值越大越好
nmi_score_1 = normalized_mutual_info_score(label_1, ac_cluster_1_pred)
print('nmi_score_1=',nmi_score_1)
#
# acc_1_lst = []
sh_score_1_lst = []
ch_score_1_lst = []
dbi_score_1_lst = []
ari_score_1_lst = []
nmi_score_1_lst = []
#
# acc_1_lst.append(acc_1)
sh_score_1_lst.append(sh_score_1)
ch_score_1_lst.append(ch_score_1)
dbi_score_1_lst.append(dbi_score_1)
ari_score_1_lst.append(ari_score_1)
nmi_score_1_lst.append(nmi_score_1)
#
df_evaluation = pd.DataFrame()
# df_evaluation['accuracy_score'] = acc_1_lst
df_evaluation['silhouette_score'] = sh_score_1_lst
df_evaluation['calinski_harabasz_score'] = ch_score_1_lst
df_evaluation['davies_bouldin_score'] = dbi_score_1_lst
df_evaluation['adjusted_rand_score'] = ari_score_1_lst
df_evaluation['normalized_mutual_info_score'] = nmi_score_1_lst
#
df_evaluation.to_csv('./embedding_eval/HA/'+path1+'df_evaluate_cluster_'+method_name+'_tSNE.csv')

# df_tsne = pd.DataFrame (X_tsne,columns = ['t_SNE1','t_SNE2'])
# df_tsne = (df_tsne - df_tsne.min()) / (df_tsne.max() - df_tsne.min())
# df_tsne ['label'] = host_name_array.tolist()

df_tsne = pd.DataFrame (X_tsne,columns = ['tSNE1','tSNE2'])
df_tsne = (df_tsne - df_tsne.min()) / (df_tsne.max() - df_tsne.min())
df_tsne ['label'] = Serotype_new_array.tolist()
#
df_all_information['tSNE1'] = df_tsne['tSNE1']
df_all_information['tSNE2'] = df_tsne['tSNE2']
df_all_information['MiniBatchKMeans_label'] = ac_cluster_1_pred
df_all_information.to_csv('./embedding_eval/HA/'+path1+'tSNE_result_with_MiniBatchKMeans_label_'+method_name+'.csv')

# plt.figure(figsize=(8, 3))
# plt.subplot(121)
# sns.scatterplot(data = df_tsne, x = 't_SNE2', y = 't_SNE1', hue = 'label', palette = color_list0[:y_num],hue_order = y_types) #
# plt.savefig('sns_scatterplot_tSNE_' + method_name + '.png', dpi = 300, bbox_inches = 'tight')

for a in range(2):
    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    sns.set(font_scale=0.3)
    sns.set_style('white')
    sns.scatterplot(data = df_tsne, x = 'tSNE2', y = 'tSNE1', hue = 'label', palette = color_list0[:y_num],hue_order = y_types) 

    plt.savefig('./embedding_eval/HA/'+path1+'sns_scatterplot_tSNE_' + method_name + '.png', dpi = 300, bbox_inches = 'tight')
np.save('./embedding_eval/HA/'+path1+'feature_' + method_name + '.npy',np_outputs,allow_pickle=True)

# plt.show()
# plt.legend(scatterpoints=1)
# plt.subplot(122)


