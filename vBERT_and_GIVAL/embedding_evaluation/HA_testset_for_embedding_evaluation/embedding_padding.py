import tensorflow as tf
# import keras as ks
import numpy as np


word_feature_len = 768
max_token_num = 185
len_max = max_token_num * word_feature_len
input_array0 = np.load('try_AIV_sample_token_onlywith_HMM_CDS_4_HA_serotype2000_38w.npy', allow_pickle=True)
name = '2000_serotype_new_sample_AIV_10w_data_38w_epoch_256cut_model_HA'
print(input_array0[0])
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
print(len(input_array))
print(len(input_array[0]))
X = tf.keras.preprocessing.sequence.pad_sequences(input_array, maxlen=None, dtype='object',padding='post')
#print(X)


np.save('feature_after_pad_seq_'+name+'.npy',X,allow_pickle=True)



