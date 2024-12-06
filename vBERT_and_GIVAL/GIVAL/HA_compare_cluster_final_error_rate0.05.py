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
import random
import math

origin_label = 'Host'
df_loc = pd.read_csv('./result/df_loc.csv')
loc_lst = df_loc['loc'].tolist()

gene = 'HA'
CDS_num = '4'
error_rate = 0.05
dict_label_num = {'Avian':0,'Human':1}
#extract_feature
method_name = gene+'_for_predict_model_compare_HA_cluster_error_rate'+str(error_rate)

#txt_file = './txt_npy_file/for_predict_model_token_4aa_CDS_'+CDS_num+'_'+gene+'.txt'
npy_file = './txt_npy_file/for_predict_model_token_onlywith_HMM_CDS_'+CDS_num+'_'+gene+'.npy'
npy_test_file = './test_set/token_onlywith_HMM_'+gene+'_test_set.npy'
#df_csv_origin = pd.read_csv('./test_set/HA_NA_test_set_host_human_avian.csv')


umap = umap.UMAP(n_neighbors=100, n_components=2, random_state=10)
pca = PCA(n_components=2, random_state=10)
tsne = TSNE(n_components=2, random_state=10)

array = np.load(npy_file, allow_pickle=True)
array_test = np.load(npy_test_file, allow_pickle=True)



#padding
dim_model = 16
word_feature_len = 768
#max_token_num = 185
#len_max = max_token_num * word_feature_len
input_array0 = array

#print(input_array0[0])
record_len_lst = []

for record_lst in input_array0:
    record_len_lst.append(len(record_lst))


maxlen_token = max(record_len_lst)
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
X = tf.keras.preprocessing.sequence.pad_sequences(input_array, maxlen=maxlen0, dtype='object',padding='post')
#print(len(X[0]))
#maxlen = len(X[0])
#print(maxlen)
maxlen = maxlen0
print('maxlen=',maxlen)
'''
input_array_test0 = array_test
input_array_test = []
for i in range(len(input_array_test0)):
    print(i)
    seq_composition = input_array_test0[i]
    seq_composition_len_reshape = len(seq_composition) * word_feature_len
    reshape_composition0 = np.reshape(seq_composition,(1,seq_composition_len_reshape))
    reshape_composition = list(reshape_composition0[0])
    input_array_test.append(reshape_composition)
#print(len(input_array))
#print(len(input_array[0]))
X_test = tf.keras.preprocessing.sequence.pad_sequences(input_array_test, maxlen=maxlen0, dtype='object',padding='post')
'''


#dimentional_reduction_and_clustering
df_all_information = pd.read_csv('./csv_file/'+gene+'_without_test_set_DCR_human_avian_sampled_with_ref.csv')
#label_1 = list(df_all_information['Host'])
#maxlen = len(input_array[0])
#for l in range(len(X)):
    #sample_composition = X[i]
    #reshape_composition0 = np.reshape(sample_composition,(1,maxlen*word_feature_len))
    #reshape_composition = reshape_composition0[0]
    #reshape_composition_lst.append(reshape_composition)
    # print(reshape_composition)
    #Serotype_new = df_all_information.loc[i,'Serotype_new']
    #Serotype_new_lst.append(Serotype_new)
df_all_information['composition_cut'] = list(X)
df_H5 = df_all_information[(df_all_information['Serotype_new_H']=='H5')]
df_H5['clade'] = ['no' for a in range(len(df_H5))]
df_H3 = df_all_information[(df_all_information['Serotype_new_H']=='H3')]
df_else = df_all_information[(df_all_information['Serotype_new_H']!='H3')&(df_all_information['Serotype_new_H']!='H5')]
df_else['clade'] = ['no' for a in range(len(df_else))]
df_ref_clade = pd.read_csv('./csv_file/ref_seq_clade_H3N2_HA.csv')
ref_seq_lst = df_ref_clade['ref_seq'].tolist()
ref_strain_name_lst = df_ref_clade['strain_name'].tolist()
ref_clade_lst = df_ref_clade['clade'].tolist()
best_clade_lst = []
#df_H3.to_csv('./csv_file/new/'+method_name+'df_H3.csv',index=False)
#df_H3 = pd.read_csv('./csv_file/new/'+method_name+'df_H3.csv')
amino_seq_lst_H3 = list(df_H3['CDS_4_amino_seq'])
for i in range(len(df_H3)):
    amino_seq = amino_seq_lst_H3[i]
    LD_lst = []
    for j in range(len(ref_seq_lst)):
        ref_seq = ref_seq_lst[j]
        LD = Levenshtein.distance(amino_seq,ref_seq)
        LD_lst.append(LD)
    LD_min = min(LD_lst)
    index_min = LD_lst.index(LD_min)
    best_clade = ref_clade_lst[index_min]
    best_clade_lst.append(best_clade)
df_H3['clade'] = best_clade_lst
df_H3_test = df_H3[df_H3['clade']=='3a.3a']
df_H3_test = df_H3_test[df_H3_test['Host']=='Human']
df_H5 = df_H5[df_H5['Host']=='Avian']
df_H3_train = df_H3[df_H3['clade']!='3a.3a']
print('df_H3_test:',len(df_H3_test))
print('df_H3_train:',len(df_H3_train))
df_else['Host1'] = df_else['Host'].tolist()

H3_train_host_lst = df_H3_train['Host'].tolist()
change_site_lst = []
change_num = math.ceil(len(H3_train_host_lst) * error_rate)
for i in range(len(H3_train_host_lst)):
    host_H3 = H3_train_host_lst[i]
    if host_H3 == 'Human':
        change_site_lst.append(i)
#sample_frac = change_num/len(change_site_lst)
random.seed(10)
sampled_change_site_lst = random.sample(change_site_lst,change_num)
Host_error_lst = []
for j in range(len(H3_train_host_lst)):
    host_H3_origin = H3_train_host_lst[j]
    if j in sampled_change_site_lst:
        Host_error_lst.append('Avian')
    else:
        Host_error_lst.append(host_H3_origin)
df_H3_train['Host1'] = Host_error_lst

df_all_information = pd.concat([df_H3_train,df_else],ignore_index=True)
df_all_information1 = pd.concat([df_H3_test,df_H5],ignore_index=True)
df_all_information1['Host1'] = df_all_information1['Host'].tolist()
#df_all_information1 = df_all_information[(df_all_information['Serotype_new_H']=='H3')|(df_all_information['Serotype_new_H']=='H5')]    
#df_all_information = df_all_information[(df_all_information['Serotype_new_H']!='H3')&(df_all_information['Serotype_new_H']!='H5')]
#df_all_information1 = df_all_information[(df_all_information['Serotype_new_H']=='H5')]    
#df_all_information = df_all_information[(df_all_information['Serotype_new_H']!='H5')]
#df_all_information1 = df_all_information[(df_all_information['Serotype_new_H']=='H1')|(df_all_information['Serotype_new_H']=='H5')]    
#df_all_information = df_all_information[(df_all_information['Serotype_new_H']!='H1')&(df_all_information['Serotype_new_H']!='H5')]

true_label_origin0 = df_all_information1['Host'].tolist()
true_label_origin = []
for label_origin0 in true_label_origin0:
    label_num0 = int(dict_label_num[label_origin0])
    true_label_origin.append(label_num0)

X = list(df_all_information['composition_cut'])
X_test = list(df_all_information1['composition_cut'])
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
#df_all_information['composition_cut'] = list(X)
df_all_information['MiniBatchKMeans_label_cut'] = labels_final_lst
host_lst = list(df_all_information[origin_label])


sil_cluster = silhouette_score(list(X),labels_final_lst)
sil_origin = silhouette_score(list(X),host_lst)
sil_cluster_lst = [sil_cluster]
sil_origin_lst = [sil_origin]
df_sil = pd.DataFrame()
df_sil['sil_cluster'] = sil_cluster_lst
df_sil['sil_origin'] = sil_origin_lst
df_sil.to_csv('./csv_file/new/'+method_name+'_sil_score.csv')

df_all_information.to_csv('./csv_file/new/'+method_name+'_with_new_label.csv')

#sample_based_on_new_label
label_lst = sorted(set(df_all_information['MiniBatchKMeans_label_cut']))
label_len_lst = []
for label in label_lst:
    df_label = df_all_information[df_all_information['MiniBatchKMeans_label_cut']==label]
    print(len(df_label))
    label_len_lst.append(len(df_label))

df_sample = pd.DataFrame()
for label0 in label_lst:
    df_label0 = df_all_information[df_all_information['MiniBatchKMeans_label_cut']==label0]
    if len(df_label0)>min(label_len_lst):
        df_label0 = sklearn.utils.shuffle(df_label0,random_state=10)
        df_label0 = df_label0.sample(frac=min(label_len_lst)/len(df_label0),random_state=10)
    df_sample = pd.concat([df_sample,df_label0],ignore_index=True)
print(len(df_sample))
MiniBatchKMeans_label_cut_lst = list(sorted(set(df_sample['MiniBatchKMeans_label_cut'])))

dict_label_cluster_host = {}
for label_cluster in MiniBatchKMeans_label_cut_lst:
    df_label_cluster = df_sample[df_sample['MiniBatchKMeans_label_cut']==label_cluster]
    cluster_host_lst = list(df_label_cluster['Host'])
    counter_host = Counter(cluster_host_lst)
    most_common_host_lst = counter_host.most_common(1)
    most_common_host = most_common_host_lst[0] 
    most_common_host = most_common_host[0] 
    dict_label_cluster_host[label_cluster] = most_common_host
print('dict_label_cluster_host=',dict_label_cluster_host)
dict_label_cluster_host_num = {}
for cluster_label in dict_label_cluster_host.keys():
    host_label = dict_label_cluster_host[cluster_label]
    host_label_num = dict_label_num[host_label]
    dict_label_cluster_host_num[cluster_label] = host_label_num   
pickle.dump(dict_label_cluster_host,open('./pkl_file/'+method_name+'dict_label_cluster_host.pkl','wb'))

#X_new = np.array(df_sample['composition_cut']) #time_consuming
#np.save(method_name+'_aligned_and_re_sampled_composition.npy', X_new, allow_pickle=True)
df_sample = sklearn.utils.shuffle(df_sample,random_state=10)
df_sample.to_csv('./csv_file/new/'+method_name+'_with_new_label_sampled_and_shuffled.csv')
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    # print(X_train.size(),X_valid.size())
    return X_train, X_valid, y_train, y_valid


dropout = 0.2
inputs = df_sample['composition_cut'].tolist()
inputs = np.array(inputs).astype(np.float64)
labels = df_sample['MiniBatchKMeans_label_cut'].tolist()
num_classes = len(set(labels))
labels = np.array(labels)
in_channel = int(maxlen * 3 / word_feature_len / 16)
print(maxlen)
print(in_channel)

inputs = torch.FloatTensor(inputs)
labels = torch.LongTensor(labels)
inputs = inputs.view(inputs.size()[0], -1, 64, 64)

#    batch_size_lst = [1000, 1100, 1150]
#    lr_lst = [1e-2, 3e-2]

batch_size_lst = [100]
lr_lst = [3e-2]

for mdl_batch_size in batch_size_lst:
    for lr in lr_lst:
        print(mdl_batch_size, lr)

        setup_seed(7)
        # Define model and optimizer
        model = ResNet34(num_classes, in_channel, dropout)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate the model
        threshold_train_acc = 0.99
        threshold_valid_acc = 0.99
        N_EPOCHS = 100
        n_splits = 3
        x = []
        skf = StratifiedKFold(n_splits=n_splits)
        xx = 0
        best_valid_acc = float('-inf')
        train_loss_lst = []
        train_acc_lst = []
        valid_acc_lst = []
        valid_f1_lst = []
        trues = []
        preds = []
        probs = []
        pos_label = 1
        for j in range(n_splits):
            train_inputs, valid_inputs, train_labels, valid_labels = get_k_fold_data(n_splits, j, inputs, labels)
            train_loader = Data.DataLoader(MyDataSet_label(train_inputs, train_labels), mdl_batch_size, True, num_workers=0)
            valid_loader = Data.DataLoader(MyDataSet_label(valid_inputs, valid_labels), mdl_batch_size, True, num_workers=0)
            for epoch in range(xx * N_EPOCHS, (xx + 1) * N_EPOCHS):
                train_loss, train_acc = train(model, train_loader, optimizer, criterion)
                pred, true, prob, valid_acc, valid_f1 = evaluate(model, valid_loader)
                print('Epoch: %2d  Train Loss: %.3f  Train Acc: %.2f Valid Acc: %.2f' % (epoch + 1, train_loss, train_acc * 100, valid_acc * 100))
                if (valid_acc > threshold_valid_acc) & (train_acc > threshold_valid_acc):
                    break
                else:
                    x.append(epoch)
                    train_loss_lst.append(train_loss)
                    train_acc_lst.append(train_acc)
                    valid_acc_lst.append(valid_acc)
                    valid_f1_lst.append(valid_f1)

            pred, true, prob, valid_acc, valid_f1 = evaluate(model, valid_loader)
            pos_prob = Positive_probs(pos_label, pred, prob)

            trues.append(true)
            preds.append(pred)
            probs.append(pos_prob)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), './resnet_classification_0619'+method_name+'.pt')

            xx += 1

#        PrintTrain(x, train_loss_lst, valid_acc_lst, './train_resnet34_0619.png')
        PrintROCPRAndCM(trues, preds, probs, pos_label,
                        './cm_resnet34_0619'+method_name, './roc_resnet34_0619'+method_name+'.png', './pr_resnet34_0619'+method_name+'.png', './roc_resnet34_0619'+method_name+'.csv', './pr_resnet34_0619'+method_name+'.csv')
        df1 = pd.DataFrame()
        df1['train_loss'] = train_loss_lst
        df1['valid_acc'] = valid_acc_lst
        df1['valid_f1'] = valid_f1_lst
        df1.to_csv('./result/train_resnet34_0619'+method_name+'.csv', index=False)

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
model = ResNet34(num_classes, in_channel)
model.load_state_dict(torch.load('./resnet_classification_0619'+method_name+'.pt'))
model.to(device)
pred, prob = test(model, loader)
pred_lst = [pred_.item() for pred_ in pred]
prob_lst = [prob_.item() for prob_ in prob]

pred_host_num_lst = []
for pred0 in pred_lst:
    pred_host_num = dict_label_cluster_host_num[pred0]
    pred_host_num_lst.append(pred_host_num)

df1 = pd.DataFrame()
df1['pred'] = pred_host_num_lst
df1['prob_cluster'] = prob_lst
df1['true_label_num'] = true_label_origin
df1['true_label'] = true_label_origin0
cm = confusion_matrix(true_label_origin, pred_host_num_lst)
print('final_cm=',cm)

df1.to_csv('./result/test_resnet34_0619'+method_name+'.csv', index=False)





