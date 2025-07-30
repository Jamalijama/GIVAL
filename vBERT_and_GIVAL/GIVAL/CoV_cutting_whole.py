import jieba4.jieba as jieba
import numpy as np
import pandas as pd
import seaborn as sns
import umap
import sklearn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, \
    normalized_mutual_info_score, accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE
import tensorflow as tf
import torch
from sklearn.model_selection import StratifiedKFold
from torch import optim

from datasets import *
from module import *
from resnet_34 import *
df_loc = pd.read_csv('./result/df_loc.csv')
loc_lst = df_loc['loc'].tolist()
virus = loc_lst[4]
gene = loc_lst[2]

unmapped_seg = ['ORF1ab']
seg_lst = ['ORF1ab','S']
cds_lst = ['ORF1ab','S']
dict_gene_CDS = dict(zip(seg_lst,cds_lst))
CDS_num = dict_gene_CDS[gene]
#n_clustered = len(loc_lst)


def lev_distance(seq0, seq1):
    '''
    Parameters
    ----------
    seq0 : String
        DESCRIPTION:
            One of the two sequences to calculate Levenshtein distance.
    seq1 : String
        DESCRIPTION:
            The other of the two sequences to calculate Levenshtein distance.

    Returns
    -------
    TYPE: int.
        DESCRIPTION:
            A Levenshtein distance of int type between two sequences.
    '''

    if len(seq0) >= len(seq1):
        seqL, seqS, seqlen = seq0, seq1, len(seq0)
    else:
        seqL, seqS, seqlen = seq1, seq0, len(seq1)

    count_arr = np.arange(seqlen + 1)

    for i in range(1, len(seqL) + 1):
        ref1 = count_arr[0]
        count_arr[0] += 1

        for j in range(1, len(seqS) + 1):
            ref2 = count_arr[j]

            if seqL[i - 1] == seqS[j - 1]:
                count_arr[j] = ref1
            else:
                count_arr[j] = min(ref1, min(count_arr[j - 1], count_arr[j])) + 1
            ref1 = ref2

    return count_arr[len(seqS)]



method_name = gene+'_for_predict_model'

sentences = []


seq_cut_lst = []
amino_num = 4
df_loc = pd.read_csv('./result/df_loc.csv')
fw1 = open('./txt_npy_file/new/for_predict_model_token_' + 'onlywith_HMM_'+gene+'.txt', 'w')
loc_lst0 = list(df_loc['loc'])

df_csv0 = pd.read_csv('./csv_file/S_without_test_set_DCR_sampled_with_ref.csv')
sentences = (df_csv0['CDS_'+gene+'_amino']).tolist()
jieba.set_dictionary("dict_empty.txt")


if loc_lst0[2] not in unmapped_seg:
    for i in range(len(sentences)):
        sentence = sentences[i]
        amino_seq_new = sentence
        seq_cut_lst.append(amino_seq_new)
        amino_seq_new = amino_seq_new.replace('~','')
        sl = jieba.lcut(amino_seq_new,HMM=True)

        CDS_token_sentence_new = ' '.join(sl)
        print(CDS_token_sentence_new, file=fw1)




else:   
    for i in range(len(sentences)):
        print(i)
        sentence = sentences[i]
        amino_seq_new = sentence
        seq_cut_lst.append(amino_seq_new)

        sl = jieba.lcut(amino_seq_new,HMM=True)

        CDS_token_sentence_new = ' '.join(sl)
        print(CDS_token_sentence_new, file=fw1)

df_csv0['cut_amino_seq'] = seq_cut_lst
df_csv0.to_csv('./csv_file/new/' + loc_lst0[2] + '_without_test_set_DCR_sampled_with_ref_with_cut_seq.csv')
fw1.close()

