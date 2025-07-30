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


df_loc = pd.read_csv('./result/df_loc.csv')
loc_lst = df_loc['loc'].tolist()

gene = loc_lst[2]

unmapped_seg = ['HA', 'NA']
seg_lst = ['NP', 'HA', 'NA']
cds_lst = ['5', '4', '6']
dict_gene_CDS = dict(zip(seg_lst,cds_lst))
CDS_num = dict_gene_CDS[gene]


batch_size = 1000
method_name00 = gene+'_for_predict_model'
df_sample  =  pd.read_csv('./csv_file/new/'+method_name00+'_with_new_label_sampled_and_shuffled.csv')

array_train_set = pickle.load(open('./pkl_file/composition_cut/train_set_composition_cut.pkl','rb'))
array_test = pickle.load(open('./pkl_file/composition_cut/test_composition_cut.pkl','rb'))


dict_label_cluster_host_num = pickle.load(open('./pkl_file/'+method_name00+'dict_label_cluster_host.pkl','rb'))
maxlen = len(array_train_set[0])
word_feature_len = 768
num_classes =  len(dict_label_cluster_host_num)
in_channel = int(maxlen * 3 / word_feature_len / 16)
model = ResNet34(num_classes, in_channel)
model.load_state_dict(torch.load('./resnet_classification_0619.pt'))
model.to(device)

##########################fc
train_feature_lst = list(array_train_set)
valid_feature_lst = list(array_test)
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
df_all_information1 = pd.DataFrame()
df_all_information1['FC_data'] = fc_data_lst_valid

df11 = pd.read_csv('./result/test_resnet34_0619.csv')
pred1 = df11.loc[0,'pred']
df_cluster = df_sample[df_sample['MiniBatchKMeans_label_cut']==pred1]
label_lst100 = list(df_cluster['Host'])
label_lst1 = list(sorted(set(list(df_cluster['Host']))))
label_count_lst00 = []
new_pred_in_cluster_lst = []
for label0 in label_lst1:
    label0_count = label_lst100.count(label0)
    label_count_lst00.append(label0_count)
if (label_count_lst00[0]>=3*label_count_lst00[1])|(label_count_lst00[1]>=3*label_count_lst00[0]):
    if label_count_lst00[0]>=3*label_count_lst00[1]:
        new_pred_in_cluster_lst.append(label_lst1[0])
    else:
        new_pred_in_cluster_lst.append(label_lst1[1])
else:

    for i in range(len(df_all_information1)):
        FC1 = df_all_information1.loc[i,'FC_data']
        pred1 = df11.loc[0,'pred']
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

df1 = pd.DataFrame()
df1['pred_host'] = new_pred_in_cluster_lst
df1.to_csv('./result/test_pred_host.csv', index=False)



