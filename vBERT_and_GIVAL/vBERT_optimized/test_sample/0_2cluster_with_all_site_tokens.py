from collections import Counter
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
# print(len(color_num_list))
color_dict = dict(zip(color_num_list, cnames.values()))
# print(color_dict)
color_list0 = list(color_dict.values())

num = 100
n_clustered = 5

txt_file = 'try_AIV_sample_token_onlywith_HMM_CDS_5_NP_human_avian1000.txt'
codon_len = 499 - 1
target_loc_lst = [1,5,16, 109, 283, 313, 319, 357]
# target_loc = 357

sentences = []
with open(txt_file, 'r+') as f:
    for line in f.readlines():
        sentences.append(line[:-1].split(' '))
print(len(sentences))


tsne = TSNE(n_components=2, random_state=7)


npy_file = 'try_AIV_sample_token_onlywith_HMM_CDS_5_NP_human_avian1000_38w.npy'
array1 = np.load(npy_file, allow_pickle=True)

array_new = []
for i in range(num):
    array_new.extend(array1[i])
array_new = np.array(array_new)
print(array_new.shape)

matrix_all_tsne = tsne.fit_transform(array_new)


for target_loc in target_loc_lst:
    print(target_loc)
    target_loc_res = []
    before = 0
    for i in range(num):
        #    print(i)
        s = sentences[i]
        current = 0
        for j in range(len(s)):
            word = s[j]
            if current <= target_loc and current + len(word) >= target_loc:
                #            print(j)
                target_loc_res.append(j + before)
                before += len(s)
                break
            else:
                current += len(word)
    print(len(target_loc_res))
    # print(target_loc_res)

    target_label = [target_loc for _ in range(num)]

    # kmeans
    method_name = '10w_256cut_38w_kmeans_' + str(target_loc) + '_' + str(n_clustered) + '_' + str(num)
    kmeans_cluster = MiniBatchKMeans(n_clusters=n_clustered, random_state=10)

    kmeans_cluster_tsne_pred = kmeans_cluster.fit_predict(matrix_all_tsne)

    df_tsne = pd.DataFrame(matrix_all_tsne, columns=['tSNE1', 'tSNE2'])
    df_tsne = (df_tsne - df_tsne.min()) / (df_tsne.max() - df_tsne.min())
    df_tsne['label'] = kmeans_cluster_tsne_pred

    df_tsne_target = df_tsne.loc[target_loc_res]
    important_site_token_labels0 = list(df_tsne_target['label'])
    important_site_token_labels = Counter(important_site_token_labels0)
    most_common_label0 = important_site_token_labels.most_common(1)
    most_common_label0 = most_common_label0[0]
    most_common_label_num = int(most_common_label0[1])
    all_num = len(important_site_token_labels0)
    ratio_one_cluster = most_common_label_num/all_num
    print(ratio_one_cluster)
    ratio_lst = []
    ratio_lst.append(ratio_one_cluster)
    df_ratio = pd.DataFrame()
    df_ratio['loc_'+str(target_loc)+'_ratio'] = ratio_lst
    df_ratio.to_csv('./NP/ratio_in_one_cluster_tSNE_' + method_name + '.csv',index=False)
    df_tsne_target['label'] = target_label


    df_tsne_all = pd.concat([df_tsne, df_tsne_target])
    y_types = sorted(set(df_tsne_all['label'].tolist()))
    print(y_types)
    y_num = len(y_types)

    sns.set(font_scale=0.3)

    sns.set_style('white')

    plt.figure(figsize=(4, 3))
    sns.scatterplot(data=df_tsne_all, x='tSNE2', y='tSNE1', hue='label', palette=color_list0[:y_num], hue_order=y_types)
    plt.savefig('./NP/sns_scatterplot_tSNE_' + method_name + '.png', dpi=300, bbox_inches='tight')
    plt.close()
    df_tsne_all.to_csv('./NP/'+method_name+'_tsne_result_with_label.csv',index=False)

   
