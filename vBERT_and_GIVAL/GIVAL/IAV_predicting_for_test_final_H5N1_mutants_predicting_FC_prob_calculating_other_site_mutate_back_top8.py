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
from scipy.spatial import distance
import math

def softmax(x):
    exp_sum = np.sum(np.exp(x))
    prob = np.exp(x)/exp_sum
    return prob


origin_label = 'Host'
df_loc = pd.read_csv('./result/df_loc.csv')
loc_lst = df_loc['loc'].tolist()
virus = loc_lst[4]
gene = loc_lst[2]
method_name = gene+'_for_predict_model'

unmapped_seg = ['HA', 'NA']
seg_lst = ['NP', 'HA', 'NA']
cds_lst = ['5', '4', '6']
dict_gene_CDS = dict(zip(seg_lst,cds_lst))
CDS_num = dict_gene_CDS[gene]
#n_clustered = len(loc_lst)
dict_label_num = {'Mutant':1}

dict_label_cluster_host= {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
dict_label_cluster_host_num= {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}


#extract_feature
method_name1 = gene+'_for_predict_model_H5N1_mutants_predicting_other_site_mutate_back_top8'
file = './test_set/test_H5N1/first_bayes_mutate_back/other_22_site_mutate_back/top_8_sites_based_on_first_bayes_30000_seq.csv'
df = pd.read_csv(file)
df['Host'] = ['Mutant' for i in range(len(df))]

npy_file = './test_set/test_H5N1/first_bayes_mutate_back/other_22_site_mutate_back/top_8_sites_based_on_first_bayes_30000_seq.npy'

array = np.load(npy_file, allow_pickle=True)

mat_cut_lst = list(array)

umap = umap.UMAP(n_neighbors=100, n_components=2, random_state=10)
pca = PCA(n_components=2, random_state=10)
tsne = TSNE(n_components=2, random_state=10)

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

df_all_information1 = df_all_information

#X = list(df_all_information['composition_cut'])

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

df_all_information1.copy()
df_all_information1['pred_dynamic_cluster'] = pred_lst
df_all_information1['pred_true_label'] = pred_host_num_lst
df_all_information1['prob_cluster'] = prob_lst


batch_size = 1000

df_sample  =  pd.read_csv('./csv_file/new/'+method_name+'_with_new_label_sampled_and_shuffled.csv')

array_train_set = pickle.load(open('./pkl_file/composition_cut/train_set_composition_cut.pkl','rb'))

train_feature_lst = list(array_train_set)
valid_feature_lst = X_test
#####train_set
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

#df_sample['pred'] = pred_lst_train
#df_sample['prob'] = prob_lst_train
print('len(df_sample)=',len(df_sample))
print('len(fc_data_lst_train)=',len(fc_data_lst_train))
df_sample['FC_data'] = fc_data_lst_train

#####valid_set
fc_data_lst_valid = []
pred_lst_valid = []
prob_lst_valid = []
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

#df_all_information1['pred'] = pred_lst_valid
#df_all_information1['prob'] = prob_lst_valid
df_all_information1['FC_data'] = fc_data_lst_valid

label_set = [iii for iii in range(num_classes)]
prob_1_lst = []
prob_2_lst = []
true_label_set = list(sorted(set(dict_label_cluster_host.values())))
empty_lst_lst = [[] for k in range(len(true_label_set))]
dict_true_flexible_label_lst = dict(zip(true_label_set.copy(),empty_lst_lst.copy()))

for flex_label in dict_label_cluster_host.keys():
    true_label = dict_label_cluster_host[flex_label]
    dict_true_flexible_label_lst[true_label].append(flex_label)
print(dict_true_flexible_label_lst)
dist_score_true_label_lst = [[] for r in range(len(true_label_set))]
dict_dist_lst = []
for i in range(len(df_all_information1)):
    print(i)
    pred_true_label = df_all_information1.loc[i,'pred_true_label']
    fc_test = np.array(list(df_all_information1.loc[i,'FC_data']))
    flexible_label_dist_lst = []
    for label in label_set:
        df_label = df_sample[df_sample['MiniBatchKMeans_label_cut']==label]
        fc_label_lst = df_label['FC_data'].tolist()
        fc_ED_lst = []
        for fc in fc_label_lst:
            fc_train = np.array(fc)
            #ED = distance.euclidean(fc_test,fc_train)
            cos_value = np.dot(fc_test,fc_train)/(np.linalg.norm(fc_test)*np.linalg.norm(fc_train))
            ED = 1-cos_value
            fc_ED_lst.append(ED)
        dist_label = np.mean(fc_ED_lst)
        flexible_label_dist_lst.append(dist_label)
    dict_label_dist = dict(zip(label_set,flexible_label_dist_lst))
   # print('dict_label_dist',dict_label_dist)
    dict_dist_lst.append(dict_label_dist)
    dict_true_label_dist = {}
    for true_label in dict_true_flexible_label_lst.keys():
        flex_label_lst = dict_true_flexible_label_lst[true_label]
        #print('flex_label_lst',flex_label_lst)
        dist_lst_per_true_label = []
        for i0 in range(len(flex_label_lst)):
            flex_label = flex_label_lst[i0]
            #print('flex_label',flex_label)
            dist_label = dict_label_dist[flex_label]
            #print('flex_label_lst',flex_label_lst)
            dist_lst_per_true_label.append(dist_label)
            #print('flex_label_lst',flex_label_lst)
        min_dist_true_label0 = np.min(dist_lst_per_true_label)

        dict_true_label_dist[true_label] = min_dist_true_label0
    #print('dict_true_label_dist',dict_true_label_dist)
    
    sum_all_dist_true_labels = np.sum(list(dict_true_label_dist.values()))
    dict_true_label_dist_ratio = {}
    for true_label in dict_true_label_dist.keys():
        dist1 = dict_true_label_dist[true_label]
        dist_ratio = 1-dist1/sum_all_dist_true_labels
        dict_true_label_dist_ratio[true_label] = dist_ratio
    dist_ratio_lst0 = list(dict_true_label_dist_ratio.values())
    dist_ratio_lst_final = softmax(dist_ratio_lst0)
    dict_true_label_dist_score_final = dict(zip(true_label_set.copy(),dist_ratio_lst_final))
    #print('dict_true_label_dist_score_final',dict_true_label_dist_score_final)
    max_prob = np.max(list(dict_true_label_dist_score_final.values()))
    if dict_true_label_dist_score_final[pred_true_label] != max_prob:
        mean_prob_value = 1/len(dict_true_label_dist_score_final)
        max_prob_value = mean_prob_value + 0.0001
        other_prob_value = (1-max_prob_value)/(len(dict_true_label_dist_score_final)-1)
        for true_label00 in dict_true_label_dist_score_final.keys():
            if true_label00 == pred_true_label:
                dict_true_label_dist_score_final[true_label00] = max_prob_value
            else:
                dict_true_label_dist_score_final[true_label00] = other_prob_value
    for kk in range(len(dict_true_label_dist_score_final)):
        dist_score = dict_true_label_dist_score_final[kk]
        dist_score_true_label_lst[kk].append(dist_score)
for j in range(len(dict_true_label_dist_score_final)):
    j_lst = dist_score_true_label_lst[j]
    df_all_information1['dist_score_true_label_'+str(j)] = j_lst     
    
df_all_information1['dict_dist'] = dict_dist_lst
df_all_information1.to_csv('./result/test_resnet34_0619_all_test'+method_name1+'_with_FC_prob_calculating.csv', index=False)
