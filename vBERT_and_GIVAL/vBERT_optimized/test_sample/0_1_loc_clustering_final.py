import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, \
    normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

cnames = {
    'lightblue': '#ADD8E6',
    'deepskyblue': '#00BFFF',
    'cadetblue': '#5F9EA0',
    'cyan': '#00FFFF',
    'purple': '#800080',
    'orchid': '#DA70D6',
    'lightgreen': '#90EE90',
    'darkgreen': '#006400',
    'yellow': '#FFFF00',
    'yellowgreen': '#9ACD32',
    'deeppink': '#FF1493',
    'burlywood': '#DEB887',
    'red': '#FF0000',
    'indianred': '#CD5C5C',
    'darkred': '#8B0000',
}

color_num_list = list(range(1, 16, 1))
# print (len(color_num_list))
color_dict = dict(zip(color_num_list, cnames.values()))
# print (color_dict)
color_list0 = list(color_dict.values())

#method_name = '10w_256cut_38w_np_diff_loc'
#
#txt_file = 'try_AIV_sample_token_onlywith_HMM_CDS_5.txt'
#npy_file = 'try_AIV_sample_token_onlywith_HMM_CDS_5_38w.npy'

method_name = '10w_256cut_38w_NP_diff_loc'

txt_file = 'try_AIV_sample_token_onlywith_HMM_CDS_5_NP_human_avian1000.txt'
npy_file = 'try_AIV_sample_token_onlywith_HMM_CDS_5_NP_human_avian1000_38w.npy'

#umap = umap.UMAP(n_neighbors=100, n_components=2, random_state=7)
pca = PCA(n_components=2, random_state=10)
#tsne = TSNE(n_components=2, random_state=7)

sentences = []
with open(txt_file, 'r') as f:
    for line in f.readlines():
        sentences.append(line[:-1].split(' '))
print(len(sentences))

loc_lst = [16, 313, 319, 357]

n_clustered = len(loc_lst)
array = np.load(npy_file, allow_pickle=True)

lst_all = []
label_all = []
for l in range(len(loc_lst)):
    res = []
    for i in range(len(sentences)):
        current = 0
        s = sentences[i]
        matrix = array[i]
        for j in range(len(s)):
            word = s[j]
            if current <= loc_lst[l] and current + len(word) >= loc_lst[l]:
                #                print(j)
                res.append(matrix[j])
                break
            else:
                current += len(word)

    res_array = np.array(res)

    label = [loc_lst[l] for _ in range(array.shape[0])]
    lst_all.extend(res_array)
    label_all.extend(label)

y_types = sorted(set(label_all))
y_num = len(y_types)

matrix_all = np.array(lst_all)



matrix_all_pca = pca.fit_transform(matrix_all)

df_pca = pd.DataFrame(matrix_all_pca, columns=['PCA1', 'PCA2'])
df_pca = (df_pca - df_pca.min()) / (df_pca.max() - df_pca.min())
df_pca['label'] = label_all
df_pca.to_csv('./NP/df_pca_'+method_name+'.csv',index=False)

sns.set(font_scale=0.3)
sns.set_style('white')

plt.figure(figsize=(4, 3))
sns.scatterplot(data=df_pca, x='PCA2', y='PCA1', hue='label', palette=color_list0[:y_num], hue_order=y_types)
plt.savefig('./NP/sns_scatterplot_PCA_' + method_name + '.png', dpi=300, bbox_inches='tight')



kmeans_cluster = MiniBatchKMeans(n_clusters=n_clustered, random_state=10)
kmeans_cluster_pca_pred = kmeans_cluster.fit_predict(matrix_all_pca)


score_lst = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score', 'adjusted_rand_score',
             'normalized_mutual_info_score']
cluster_label_lst = []
pca_score_lst = []

# pca

sh_score = silhouette_score(matrix_all_pca, kmeans_cluster_pca_pred, metric='euclidean')
ch_score = calinski_harabasz_score(matrix_all_pca, kmeans_cluster_pca_pred)
dbi_score = davies_bouldin_score(matrix_all_pca, kmeans_cluster_pca_pred)
ari_score = adjusted_rand_score(label_all, kmeans_cluster_pca_pred)
nmi_score = normalized_mutual_info_score(label_all, kmeans_cluster_pca_pred)
pca_score_lst.append(sh_score)
pca_score_lst.append(ch_score)
pca_score_lst.append(dbi_score)
pca_score_lst.append(ari_score)
pca_score_lst.append(nmi_score)




df1 = pd.DataFrame()
df1['label'] = cluster_label_lst
df1['pca'] = pca_score_lst

df1.to_csv('./NP/cluster_diff_loc' + method_name + '.csv', index=False)
