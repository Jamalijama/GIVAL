import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

gene = 'HA'
CDS_num = '4'
method_name = gene+'_for_predict_model_compare_HA_host'
#df_ = pd.read_csv('./csv_file/new/'+method_name+'_with_new_label_sampled_and_shuffled.csv')
df_ = pd.read_csv('./csv_file/new/'+method_name+'_with_new_label_all_for_SSE_calculation.csv')
label_host = df_['Host']
label_dynamic = df_['MiniBatchKMeans_label_cut']
data = df_[['PCA1','PCA2']]
sil_coef_host = silhouette_score(data,label_host)
sil_coef_dynamic = silhouette_score(data,label_dynamic)
print(sil_coef_host)
print(sil_coef_dynamic)



