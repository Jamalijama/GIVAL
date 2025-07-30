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
from collections import Counter

df_loc = pd.read_csv('./result/df_loc.csv')
loc_lst = df_loc['loc'].tolist()
virus = loc_lst[-1]
gene = loc_lst[2]

method_name = virus+'_'+gene+'_for_predict_model'


df_sample = pd.read_csv('./csv_file/new/'+method_name+'_with_new_label_sampled_and_shuffled.csv')


df1 = pd.read_csv('./result/test_resnet34_0619.csv')
df1_pred = df1.loc[0,'pred']

df_cluster = df_sample[df_sample['MiniBatchKMeans_label_cut']==df1_pred]
host_lst = list(df_cluster['host'])
counts = Counter(host_lst)
count0 = list(counts.keys())[0]
pred_host_lst = [count0]
df111 = pd.DataFrame()
df111['pred_host'] = pred_host_lst
df111.to_csv('./result/test_pred_host.csv', index=False)




