import gensim
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import pickle

#CBOW模型：
#sentence为语料库；
#size为向量维度；
#window为上下文大小窗口；
#sample为采样率；
#min_count是最低出现数，默认数值是5；
#size是gensim Word2Vec将词汇映射到的N维空间的维度数量（N）默认的size数是100；
#iter是模型训练时在整个训练语料库上的迭代次数，假如参与训练的文本量较少，就需要把这个参数调大一些。iter的默认值为5；
#sg是模型训练所采用的的算法类型：1 代表 skip-gram，0代表 CBOW，sg的默认值为0；
#window控制窗口，如果设得较小，那么模型学习到的是词汇间的组合性关系（词性相异）；如果设置得较大，会学习到词汇之间的聚合性关系（词性相同）。模型默认的window数值为5；

dataset = pickle.load(open('try_AIV_sample_token_onlywith_HMM_CDS_5_NP_human_avian1000.pkl','rb'))
model = Word2Vec(sentences=dataset,vector_size= 768,window = 5,min_count = 0,workers = 4,sample = 0.001,sg = 0)
#获取单词向量
matrix_word2vec_composition = []
for sequence in dataset:
    seq_word2vec_composition = []
    for codon in sequence:
        word_vec_codon = model.wv[codon]
        seq_word2vec_composition.append(word_vec_codon)
    matrix_word2vec_composition.append(seq_word2vec_composition)

with open('NP_word2vec_HMM_feature.pkl','wb') as f2:
    pickle.dump(matrix_word2vec_composition,f2)


# 保存模型
model.save('NP_word2vec_HMM.model')


