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
#from imblearn.over_sampling import SMOTE

maxlen=185
word_dim=768
gene = 'HA'
method_name = '2000sample_serotype_new_AIV_HA_10w_data_38w_epoch_256cut_model_minibatch_kmeans_pad_0'
matrix_npy = np.load('feature_after_pad_seq_2000_serotype_new_sample_AIV_10w_data_38w_epoch_256cut_model_HA.npy', allow_pickle=True)

df_all_information = pd.read_csv('HA_sampled_Serotype_new_not_in_training_set_and_1wHA.csv')
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
df_evaluation.to_csv('df_evaluate_cluster_'+method_name+'_tSNE.csv')

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
df_all_information.to_csv('tSNE_result_with_MiniBatchKMeans_label_'+method_name+'.csv')

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

    plt.savefig('sns_scatterplot_tSNE_' + method_name + '.png', dpi = 300, bbox_inches = 'tight')


# plt.show()
# plt.legend(scatterpoints=1)
# plt.subplot(122)


