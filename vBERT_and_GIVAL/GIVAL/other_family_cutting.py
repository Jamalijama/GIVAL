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

def count_AApair (seq, Num0 = 400, freq = True):
    amino_table = ['I', 'D', 'M', 'H', 'E', 'W', 'R', 'L', 'Y', 'Q', 'G', 'A', 'S', 'P', 'C', 'T', 'V', 'F', 'N', 'K']
    AApair_table = [AA1 + AA2 for AA1 in amino_table for AA2 in amino_table]

    seq_len = len(seq)
    seq = seq.upper()
    count_AApair = np.zeros(Num0)
    for i in range(0, seq_len-1, 1):
        cut = seq[i:i+2]
        if cut in AApair_table:
            AA = AApair_table.index(cut)
            print(AA)
            count_AApair[AA] += 1

    if freq:
        return np.hstack(count_AApair/seq_len)
    else:
        return np.hstack(count_AApair)



df_loc = pd.read_csv('./result/df_loc.csv')
df_loc = df_loc['loc'].tolist()
family = df_loc[4]
gene = df_loc[2]

unmapped_seg = [gene]
seg_lst = [gene]
cds_lst = [gene]
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
fw1 = open('./txt_npy_file/new/for_predict_model_token_' + 'onlywith_HMM.txt', 'w')
loc_lst0 = list(df_loc['loc'])

df_csv00 = pd.read_csv('./csv_file/seq_all_family.csv')
df_csv01 = pd.read_csv('./csv_file/df_all_mpox.csv')
df_csv0 = pd.concat([df_csv00,df_csv01],ignore_index=True)
print(family)
print(len(df_csv0))
df_csv0 = df_csv0[df_csv0['family']==family]
print(gene)
print(len(df_csv0))
df_csv0 = df_csv0[df_csv0['protein']==gene]
print(len(df_csv0))

if len(df_csv0) == 0:
    df_mapped_ref = pd.read_csv('./result/mapped_ref_seq.csv')
    mapped_ref_lst = list(df_mapped_ref['mapped_ref'])
    mapped_ref = str(mapped_ref_lst[0])
    mapped_ref_len = len(mapped_ref)
    mapped_ref_AApair_vec = count_AApair(mapped_ref)
    df_csv02 = pd.read_csv('../../data/file/df_all_after_deduplication.csv')
    df_csv02 = df_csv02[df_csv02['amino_seq_len_new']>=0.9*mapped_ref_len]
    df_csv02 = df_csv02[df_csv02['amino_seq_len_new']<=1.1*mapped_ref_len]
    df_csv02.to_csv('./csv_file/for_reindex.csv',index=False)
    df_csv02 = pd.read_csv('./csv_file/for_reindex.csv',index=False)
    AApair_vec_lst = [mapped_ref_AApair_vec]
    for a in range(len(df_csv02)):
        seq_aa = str(df_csv02.loc[a,'amino_seq_new'])
        AApair_vec = count_AApair(seq_aa,freq=True)
        AApair_vec_lst.append(AApair_vec)
    AApair_vec_lst = np.array(AApair_vec_lst)
    pca = PCA(n_components=2, random_state=10)
    X_pca = pca.fit_transform(AApair_vec_lst)
    df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    df_pca = (df_pca - df_pca.min()) / (df_pca.max() - df_pca.min())

    # select_K_best
    SSE_lst = []
    for k in range(1, 12):
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=10).fit(df_pca)
        SSE_lst.append(kmeans.inertia_)

    SSE_interval_ratio_lst = []
    for i in range(1, len(SSE_lst) - 1):
        SSE_i_before = SSE_lst[i - 1]
        SSE_i = SSE_lst[i]
        SSE_i_after = SSE_lst[i + 1]
        SSE_interval_before = SSE_i_before - SSE_i
        SSE_interval_after = SSE_i - SSE_i_after
        SSE_interval_ratio = SSE_interval_before / SSE_interval_after
        SSE_interval_ratio_lst.append(SSE_interval_ratio)
    for i0 in range(len(SSE_interval_ratio_lst)):
        SSE_inter_rat = SSE_interval_ratio_lst[i0]
        if SSE_inter_rat == max(SSE_interval_ratio_lst):
            k_best = i0 + 2
            break
    print(k_best)

    k_means_final = MiniBatchKMeans(n_clusters=k_best, random_state=10)
    labels_final_lst = k_means_final.fit_predict(df_pca)

    label_mapped_ref = labels_final_lst[0]
    label_other_lst = list(labels_final_lst[1:])
    df_csv02['MiniBatchKMeans_label_AApair'] = label_other_lst
    df_csv02 = df_csv02[df_csv02['MiniBatchKMeans_label_AApair']==label_mapped_ref]
    if len(df_csv02) >= 1000:
        df_csv02 = sklearn.utils.shuffle(df_csv02,random_state=10)
        df_csv02 = df_csv02.sample(n=1000,random_state=10)
    df_csv02.to_csv('./csv_file/mapped_virus_dataset.csv',index=False)
    df_csv0 = pd.read_csv('./csv_file/mapped_virus_dataset.csv')


sentences = (df_csv0['cds_seq']).tolist()
jieba.set_dictionary("dict_empty.txt")
#sentences = sentences[:5]

if loc_lst0[2] not in unmapped_seg:
    loc_start = int(loc_lst0[0])
    loc_end = int(loc_lst0[1])
    loc_lst = [loc_start, loc_end]#start_from_0
    
    for i in range(len(sentences)):
        sentence = sentences[i]
        amino_seq_new = sentence[loc_start:loc_end]
        seq_cut_lst.append(amino_seq_new)

        sl = jieba.lcut(amino_seq_new,HMM=True)

        CDS_token_sentence_new = ' '.join(sl)
        print(CDS_token_sentence_new, file=fw1)




else:
    #ref_seq_lst = []
    #indexx = seg_lst.index(loc_lst0[2])
    #file = './csv_file/' + loc_lst0[2] + '_without_test_set_DCR_sampled_with_ref.csv'
    df = df_csv0
    ref_seq_lst = df['cds_seq'].tolist()
    #ref_seq_lst = ref_seq_lst[:5]
    target_seq = loc_lst0[3]
    
    shortcut = 20
    print(len(ref_seq_lst), len(sentences))

    ref_lev_res = [[] for _ in range(len(ref_seq_lst))]
    
    for i, ref in enumerate(ref_seq_lst):
        start_cut = target_seq[:shortcut]
        end_cut = target_seq[-shortcut:]
        # print(len(start_cut), len(end_cut))
        print(i)
        start_lev_lst = []
        end_lev_lst = []
        for j in range(0, len(ref) - shortcut, 1):
            start_lev_lst.append(lev_distance(start_cut, ref[j:j + shortcut]))
            #end_lev_lst.append(lev_distance(end_cut, ref[j:j + shortcut]))
        start = start_lev_lst.index(min(start_lev_lst))
        #end = end_lev_lst.index(min(end_lev_lst))
        # print(len(target_seq), len(ref[start:end + shortcut]))
#        ref_lev_dis = lev_distance(target_seq, ref[start:end + shortcut])
        ref_lev_res[i].append(start)
        #ref_lev_res[i].append(end + shortcut)
#        ref_lev_res[i].append(ref_lev_dis)
        end = start + len(target_seq)
        end_last_short_cut = ref[end-shortcut:end]
        if lev_distance(end_cut, end_last_short_cut) < 6:
            ref_lev_res[i].append(end)
        else:
            end_lev_lst1 = []
            for k in range(-5,6):
                end_last_short_cut1 = ref[end-shortcut+k:end+k]
                end_lev_lst1.append(lev_distance(end_cut, end_last_short_cut1))
            end1 = end + end_lev_lst1.index(min(end_lev_lst1)) - 5
            ref_lev_res[i].append(end1)

    #print(ref_lev_res)
        
    for i in range(len(sentences)):
        print(i)
        current = 0
        sentence = sentences[i]
        
        loc_start = int(ref_lev_res[i][0])
        loc_end = int(ref_lev_res[i][1])
        amino_seq_new = sentence[loc_start:loc_end]
        seq_cut_lst.append(amino_seq_new)

        sl = jieba.lcut(amino_seq_new,HMM=True)

        CDS_token_sentence_new = ' '.join(sl)
        print(CDS_token_sentence_new, file=fw1)
#df_csv0 = df_csv0[:5]
df_csv0['cut_amino_seq'] = seq_cut_lst
df_csv0.to_csv('./csv_file/new/seq_all_family_with_cut.csv')
fw1.close()

