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


import pickle
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
from collections import Counter
from datasets import *
from module import *
from resnet_34 import *
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_curve, auc, \
    precision_recall_curve, average_precision_score
import Levenshtein

origin_label = 'Host'
df_loc = pd.read_csv('./result/df_loc.csv')
loc_lst = df_loc['loc'].tolist()

gene = loc_lst[2]

unmapped_seg = ['HA', 'NA']
seg_lst = ['NP', 'HA', 'NA']
cds_lst = ['5', '4', '6']
dict_gene_CDS = dict(zip(seg_lst,cds_lst))
CDS_num = dict_gene_CDS[gene]
#n_clustered = len(loc_lst)
dict_label_num = {'Mammal':1,'Unknown':2,'Swine':3}


dict_label_cluster_host_num= {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}

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


#extract_feature
method_name = gene+'_for_predict_model_other_host_predicting'

txt_file = './test_set/test_canine/token_onlywith_HMM_other_host.txt'
npy_file = './test_set/test_canine/token_onlywith_HMM_other_host.npy'

umap = umap.UMAP(n_neighbors=100, n_components=2, random_state=10)
pca = PCA(n_components=2, random_state=10)
tsne = TSNE(n_components=2, random_state=10)

sentences = []
with open(txt_file, 'r') as f:
    for line in f.readlines():
        sentences.append(line[:-1].split(' '))
print(len(sentences))

#loc_lst = [16, 109, 283, 313, 319, 357]
#loc_lst = [55, 103]#start_from_1
#loc_lst = [37, 166]#start_from_1
array = np.load(npy_file, allow_pickle=True)

mat_cut_lst = []
#df_loc = pd.read_csv('./result/df_'+gene+'_0619.csv')
loc_lst0 = list(df_loc['loc'])
if loc_lst0[2] not in unmapped_seg:
    loc_start = int(loc_lst0[0]) + 1
    loc_end = int(loc_lst0[1])
    loc_lst = [loc_start, loc_end]#start_from_1
    
    for i in range(len(sentences)):
        current = 0
        s = sentences[i]
        matrix = array[i]
        for j in range(len(s)):
            word = s[j]
            if current < loc_lst[0] and current + len(word) >= loc_lst[0]:
                j_start = j
                #breakimport tensorflow as tf
            elif current < loc_lst[1] and current + len(word) >= loc_lst[1]:
                j_end = j
                mat_cut = matrix[j_start:(j_end+1)]
                mat_cut_lst.append(mat_cut)
                #print(j_start,'_',j_end)
                break
            #else:
            current += len(word)
else:
    ref_seq_lst = []
    indexx = seg_lst.index(loc_lst0[2])
    file = './test_set/test_canine/new_AIV_HA_NA_selected_other_host_except_human_avian.csv'
    df = pd.read_csv(file)
    ref_seq_lst = df['CDS_' + cds_lst[indexx] + '_amino_seq'].tolist()
    
    target_seq = loc_lst0[3]
    print(target_seq)
    #target_seq = 'SSSDNGTCYPGDFIDYEELREQLSSVSSFERFEIFPKTSSWPNHDSNKGVTAACPHAGAKSFYKNLIWLVKKGNSYPKLSKSYINDKGKEVLVLWGIHHPSTSADQQSLYQNADAYVFVGTSRYSKKFKPEIAIRPKVRDREGRMNYYWTL'

    shortcut = 20
    print(len(ref_seq_lst), len(sentences))

    ref_lev_res = [[] for _ in range(len(ref_seq_lst))]
    cut_seq_lst = []
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
        cut_seq = ref[start:end1]
        cut_seq_lst.append(cut_seq)
    for i in range(len(sentences)):
        print(i)
        current = 0
        s = sentences[i]
        matrix = array[i]
        loc_start = int(ref_lev_res[i][0]) + 1
        loc_end = int(ref_lev_res[i][1])
        loc_lst = [loc_start, loc_end]
        for j in range(len(s)):
            word = s[j]
            if current < loc_lst[0] and current + len(word) >= loc_lst[0]:
                j_start = j
                #breakimport tensorflow as tf
            elif current < loc_lst[1] and current + len(word) >= loc_lst[1]:
                j_end = j
                mat_cut = matrix[j_start:(j_end+1)]
                mat_cut_lst.append(mat_cut)
                #print(j_start,'_',j_end)
                break
            #else:
            current += len(word)

#padding
dim_model = 16
word_feature_len = 768
#max_token_num = 185
#len_max = max_token_num * word_feature_len
input_array0 = mat_cut_lst
#input_array0_test = mat_cut_lst_test
#print(input_array0[0])
record_len_lst = []
record_len_lst_test = []
for record_lst in input_array0:
    record_len_lst.append(len(record_lst))
#for record_lst_test in input_array0_test:
   # record_len_lst_test.append(len(record_lst_test))
record_len_lst_all = record_len_lst
maxlen_token = max(record_len_lst_all)
a1 = int(maxlen_token/dim_model)
a2 = maxlen_token%dim_model
if a2 == 0:
    maxlen0 = maxlen_token*word_feature_len
else:
    maxlen0 = (a1+1)*dim_model*word_feature_len

#name = '2000_serotype_new_sample_AIV_10w_data_38w_epoch_256cut_model_HA'
#print(input_array0[0])
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
#print(len(input_array))
#print(len(input_array[0]))
maxlen0 = 49152
X = tf.keras.preprocessing.sequence.pad_sequences(input_array, maxlen=maxlen0, dtype='object',padding='post')
#print(len(X[0]))
#maxlen = len(X[0])
#print(maxlen)
maxlen = maxlen0
print(maxlen)

df_all_information = df
df_all_information['composition_cut'] = list(X)
df_all_information['cut_seq'] = cut_seq_lst
df_all_information1 = df_all_information
#df_all_information1 = df_all_information[(df_all_information['Serotype_new_H']=='H3')|(df_all_information['Serotype_new_H']=='H5')
    
#df_all_information = df_all_information[(df_all_information['Serotype_new_H']!='H3')&(df_all_information['Serotype_new_H']!='H5')]
#df_all_information1 = df_all_information[(df_all_information['Serotype_new_H']=='H5')]
    
#df_all_information = df_all_information[(df_all_information['Serotype_new_H']!='H5')]

serotype0 = df_all_information1['Serotype'].tolist()
strain_name0 = df_all_information1['Strain_name'].tolist()
continent0 = df_all_information1['Continent'].tolist()
location0 = df_all_information1['location'].tolist()
year0 = df_all_information1['Year'].tolist()
nt_seq0 = df_all_information1['CDS_4_nt_seq'].tolist()
amino_seq0 = df_all_information1['CDS_4_amino_seq'].tolist()
cut_seq0 = df_all_information1['cut_seq'].tolist()
true_label_origin0 = df_all_information1['Host'].tolist()
true_label_origin = []
for label_origin0 in true_label_origin0:
    label_num0 = int(dict_label_num[label_origin0])
    true_label_origin.append(label_num0)

X = list(df_all_information['composition_cut'])
X_pca = pca.fit_transform(X)
df_pca = pd.DataFrame (X_pca,columns = ['PCA1','PCA2'])
df_pca = (df_pca - df_pca.min()) / (df_pca.max() - df_pca.min())
# select_K_best
SSE_lst = []
for k in range(1,12):
    # print(k)
    kmeans = MiniBatchKMeans(n_clusters=k,random_state=10).fit(df_pca)
    SSE_lst.append(kmeans.inertia_)
# print((SSE_lst))
#plt.xlabel = "n_clusters"
#plt.ylabel = "SSE"
#plt.plot(range(1,12),SSE_lst,"o-")
#plt.savefig(name+'_SSE_minibatchKmeans.png', dpi = 300, bbox_inches = 'tight')
# plt.show()
SSE_interval_ratio_lst = []
for i in range(1,len(SSE_lst)-1):
    SSE_i_before = SSE_lst[i-1]
    SSE_i = SSE_lst[i]
    SSE_i_after = SSE_lst[i+1]
    SSE_interval_before = SSE_i_before-SSE_i
    SSE_interval_after = SSE_i-SSE_i_after
    SSE_interval_ratio = SSE_interval_before/SSE_interval_after
    SSE_interval_ratio_lst.append(SSE_interval_ratio)
for i0 in range(len(SSE_interval_ratio_lst)):
    SSE_inter_rat = SSE_interval_ratio_lst[i0]
    if SSE_inter_rat == max(SSE_interval_ratio_lst):
        k_best = i0+2
        break
print(k_best)
#print(SSE_lst)
#print(SSE_interval_ratio_lst)
k_means_final = MiniBatchKMeans(n_clusters=k_best, random_state=10)
labels_final_lst = k_means_final.fit_predict(df_pca)
#kmeans = k_means_final.fit(df_pca)
#cluster_centers = kmeans.cluster_centers_

df_all_information['MiniBatchKMeans_label_cut'] = labels_final_lst
host_lst = list(df_all_information[origin_label])

df_all_information['PCA1'] = df_pca['PCA1'].tolist()
df_all_information['PCA2'] = df_pca['PCA2'].tolist()
df_all_information.to_csv('./csv_file/new/'+method_name+'_with_new_label.csv')

X_test = list(df_all_information1['composition_cut'])
tests = list(X_test)
tests = np.array(tests).astype(np.float64)
tests = torch.FloatTensor(tests)
print(tests.shape)
# ids = torch.Tensor(ids)

tests = tests.view(tests.size()[0], -1, 64, 64)
mdl_batch_size = 1000
setup_seed(7)
loader = Data.DataLoader(MyDataSet_freq(tests), mdl_batch_size, False, num_workers=0)
# load the model
num_classes =  len(dict_label_cluster_host_num)
in_channel = int(maxlen * 3 / word_feature_len / 16)
model = ResNet34(num_classes, in_channel)
model.load_state_dict(torch.load('./resnet_classification_0619.pt'))
model.to(device)
pred, prob = test(model, loader)
pred_lst = [pred_.item() for pred_ in pred]
prob_lst = [prob_.item() for prob_ in prob]

pred_host_num_lst = []
for pred0 in pred_lst:
    pred_host_num = dict_label_cluster_host_num[pred0]
    pred_host_num_lst.append(pred_host_num)

df1 = pd.DataFrame()
df1['pred_dynamic_cluster'] = pred_lst
df1['pred'] = pred_host_num_lst
df1['prob_cluster'] = prob_lst
df1['true_label_num'] = true_label_origin
df1['true_label'] = true_label_origin0
df1['serotype'] = serotype0
df1['continent'] = continent0
df1['year'] = year0
df1['strain_name'] = strain_name0
df1['nt_seq'] = nt_seq0
df1['amino_seq'] = amino_seq0
df1['cut_seq'] = cut_seq0
df1['location'] = location0
cm = confusion_matrix(true_label_origin, pred_host_num_lst)
print('final_cm=',cm)
X = df_all_information1['composition_cut'].tolist()
X_pca = pca.fit_transform(X)
df_pca = pd.DataFrame (X_pca,columns = ['PCA1','PCA2'])
df_pca = (df_pca - df_pca.min()) / (df_pca.max() - df_pca.min())
df1['PCA1'] = df_pca['PCA1'].tolist()
df1['PCA2'] = df_pca['PCA2'].tolist()
df1.to_csv('./result/test_resnet34_0619_all_test'+method_name+'.csv', index=False)



