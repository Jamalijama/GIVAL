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

origin_label = 'Serotype'
df_loc = pd.read_csv('./result/df_loc.csv')
loc_lst = df_loc['loc'].tolist()

gene = loc_lst[2]

unmapped_seg = ['HA', 'NA']
seg_lst = ['NP', 'HA', 'NA']
cds_lst = ['5', '4', '6']
dict_gene_CDS = dict(zip(seg_lst,cds_lst))
CDS_num = dict_gene_CDS[gene]
#n_clustered = len(loc_lst)
dict_label_num = {'H3N2':1,'H1N1':2}


dict_label_cluster_host_num= {0: 1, 1: 0, 2: 1, 3: 1}

batch_size = 1000
method_name00 = 'HA_for_predict_model'
df_sample  =  pd.read_csv('./csv_file/new/'+method_name00+'_with_new_label_sampled_and_shuffled.csv')

array_train_set = pickle.load(open('./pkl_file/composition_cut/train_set_composition_cut.pkl','rb'))

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
method_name = gene+'_for_predict_model_pandemics'

txt_file = './test_set/pandemics/H1N1_2009_H3N2_1968.txt'
npy_file = './test_set/pandemics/H1N1_2009_H3N2_1968.npy'

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
    file = './test_set/pandemics/H1N1_2009_H3N2_1968.csv'
    df = pd.read_csv(file)
    ref_seq_lst = df['seq'].tolist()
    
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
            end1 = end
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
maxlen0 = len(array_train_set[0])
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
strain_name0 = df_all_information1['seq_id'].tolist()
amino_seq0 = df_all_information1['seq'].tolist()
cut_seq0 = df_all_information1['cut_seq'].tolist()
true_label_origin0 = df_all_information1['Serotype'].tolist()
true_label_origin = []
for label_origin0 in true_label_origin0:
    label_num0 = int(dict_label_num[label_origin0])
    true_label_origin.append(label_num0)

X = list(df_all_information['composition_cut'])

X_test = list(df_all_information1['composition_cut'])
tests = list(X_test)
test1111111 = tests
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
df1['pred_origin'] = pred_lst
df1['pred'] = pred_host_num_lst
df1['prob_cluster'] = prob_lst
df1['serotype'] = serotype0
df1['seq_id'] = strain_name0
df1['amino_seq'] = amino_seq0
df1['cut_seq'] = cut_seq0
df_all_information1['pred_origin'] = pred_lst
df_all_information1['pred'] = pred_host_num_lst
df_all_information1['prob_cluster'] = prob_lst

##########################fc
train_feature_lst = list(array_train_set)
valid_feature_lst = test1111111
#####train_set
batch_size=1000
fc_data_lst_train = []
pred_lst_train = []
prob_lst_train = []
batch_num_train = int(len(train_feature_lst)/batch_size)+1
if batch_num_train < 1:
    batch_num_train = 1
maxlen = len(train_feature_lst[0])
#num_classes = len(set(labels))
for i in range(batch_num_train):
    all_lst1 = train_feature_lst[batch_size*i:batch_size*(i+1)]
    X_test = all_lst1
    tests = list(X_test)
    tests = np.array(tests).astype(np.float64)
    tests = torch.FloatTensor(tests)
    print(tests.shape)
    # ids = torch.Tensor(ids)
    mdl_batch_size = batch_size
    tests = tests.view(tests.size()[0], -1, 64, 64)
    word_feature_len = 768
    setup_seed(7)
    loader = Data.DataLoader(MyDataSet_freq(tests), mdl_batch_size, False, num_workers=0)
    # load the model
    #num_classes =  4
    in_channel = int(maxlen * 3 / word_feature_len / 16)
    #model = ResNet34(num_classes, in_channel)
    #model.load_state_dict(torch.load('./model/new/resnet_classification_'+method_name+'.pt'))
    #model.to(device)
    pred, prob = test(model, loader)
    pred_lst = [pred_.item() for pred_ in pred]
    prob_lst = [prob_.item() for prob_ in prob]
    f2 = open('FC_data_test.pkl','rb')
    fc_data1 = pickle.load(f2)
    fc_data_lst_train.extend(fc_data1)
    pred_lst_train.extend(pred_lst)
    prob_lst_train.extend(prob_lst)
print(len(fc_data_lst_train))
print(len(pred_lst_train))
print(len(prob_lst_train))
print(len(fc_data_lst_train[0]))

#df_sample['pred'] = pred_lst_train
#df_sample['prob'] = prob_lst_train
print('len(df_sample)=',len(df_sample))
print('len(fc_data_lst_train)=',len(fc_data_lst_train))
df_sample['FC_data'] = fc_data_lst_train

#####valid_set
fc_data_lst_valid = []
pred_lst_valid = []
prob_lst_valid = []
#df_all_information1 = df1.copy()
batch_num_valid = int(len(valid_feature_lst)/batch_size)+1
if batch_num_valid < 1:
    batch_num_valid = 1
maxlen = len(valid_feature_lst[0])
#num_classes = len(set(labels))
for i in range(batch_num_valid):
    all_lst1 = valid_feature_lst[batch_size*i:batch_size*(i+1)]
    X_test = all_lst1
    tests = list(X_test)
    tests = np.array(tests).astype(np.float64)
    tests = torch.FloatTensor(tests)
    print(tests.shape)
    # ids = torch.Tensor(ids)
    mdl_batch_size = batch_size
    tests = tests.view(tests.size()[0], -1, 64, 64)
    word_feature_len = 768
    setup_seed(7)
    loader = Data.DataLoader(MyDataSet_freq(tests), mdl_batch_size, False, num_workers=0)
    # load the model
    #num_classes =  4
    in_channel = int(maxlen * 3 / word_feature_len / 16)
    #model = ResNet34(num_classes, in_channel)
    #model.load_state_dict(torch.load('./model/new/resnet_classification_'+method_name+'.pt'))
    #model.to(device)
    pred, prob = test(model, loader)
    pred_lst = [pred_.item() for pred_ in pred]
    prob_lst = [prob_.item() for prob_ in prob]
    f2 = open('FC_data_test.pkl','rb')
    fc_data1 = pickle.load(f2)
    fc_data_lst_valid.extend(fc_data1)
    pred_lst_valid.extend(pred_lst)
    prob_lst_valid.extend(prob_lst)
print(len(fc_data_lst_valid))
print(len(pred_lst_valid))
print(len(prob_lst_valid))
print(len(fc_data_lst_valid[0]))

#df_all_information1['pred'] = pred_lst_valid
#df_all_information1['prob'] = prob_lst_valid
df_all_information1['FC_data'] = fc_data_lst_valid



new_pred_in_cluster_lst = []
for i in range(len(df_all_information1)):
    FC1 = df_all_information1.loc[i,'FC_data']
    pred1 = df_all_information1.loc[i,'pred_origin']
    df_cluster = df_sample[df_sample['MiniBatchKMeans_label_cut']==pred1]
    label_lst1 = list(sorted(set(list(df_cluster['Host']))))
    dict_label_dist = {}
    for label1 in label_lst1:
        df_label_cluster = df_cluster[df_cluster['Host']==label1]
        FC_label_cluster_lst = list(df_label_cluster['FC_data'])
        FC_dist_label_cluster_lst = []
        for FC2 in FC_label_cluster_lst:
            cos_value = np.dot(FC1,FC2)/(np.linalg.norm(FC1)*np.linalg.norm(FC2))
            FC_dist_label_cluster_lst.append(1-cos_value)
        dict_label_dist[label1] = np.min(FC_dist_label_cluster_lst)
    min_dist = np.min(list(dict_label_dist.values()))
    min_dist_label_lst = []
    for label2 in dict_label_dist.keys():
        if dict_label_dist[label2] == min_dist:
            min_dist_label_lst.append(label2)
    if len(min_dist_label_lst) == 1:
        new_pred_in_cluster_lst.append(min_dist_label_lst[0])
    else:
        new_pred_in_cluster_lst.append('both')


df1['new_pred_in_cluster'] = new_pred_in_cluster_lst
df1.to_csv('./result/test_resnet34_0619_all_test'+method_name+'.csv', index=False)



