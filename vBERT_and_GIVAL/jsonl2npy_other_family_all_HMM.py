import json
import pandas as pd
import numpy as np
import sys

#input_file = sys.argv[1]
#output_file = sys.argv[2]
#file_lst = ['try_AIV_sample_token_onlywith_HMM_CDS_4_HA_', 'try_AIV_sample_token_onlywith_HMM_CDS_4_HA_serotype_',
#          'try_AIV_sample_token_onlywith_HMM_CDS_5_']
dirr = './GIVAL/txt_npy_file/new/'
#file_lst = ['token_4aa_CDS_5_NP_', 'token_onlywith_HMM_CDS_5_NP_']
#file_end = '38w'
file_lst = ['for_predict_model_token_onlywith_HMM']
file_end = ''
for f in file_lst:
    input_file = f + file_end + '.jsonl'
    print(input_file)
    output_file = input_file[:-6] + '.npy'
    
    with open(dirr + input_file, 'r') as json_file:
        json_list = list(json_file)
    print(len(json_list))
    
    max_seq_len = 256
    
    #fout = open(output_file, 'w')
    sentence_lst = []
    for json_str in json_list:
        tokens = json.loads(json_str)["features"]
    #    print(len(tokens))
        tokens_lst = []
        for token in tokens:
            if token['token'] in ['[CLS]', '[SEP]']:
                continue
            else:
                last_layers = np.sum([
                    token['layers'][0]['values'],
                    token['layers'][1]['values'],
                    token['layers'][2]['values'],
                    token['layers'][3]['values'],
                ], axis=0)
    #            print(last_layers.shape)
                tokens_lst.append(last_layers)
    #            fout.write(f'{",".join(["{:f}".format(i) for i in last_layers])}\n')
    #    fout.write(f'\n')
#        print(len(tokens_lst))
    #    for i in range(len(tokens_lst), max_seq_len):
    #        tokens_lst.append(np.zeros(768))
    #    print(len(tokens_lst))
        sentence_lst.append(tokens_lst)
        
    sarray = np.array([np.array(s) for s in sentence_lst])
    print(sarray.shape)
    #print(sarray[0][1])
    #print(sarray[3][1])
    np.save(dirr + output_file, sarray, allow_pickle=True)

#f = open('input.txt')
#lines = f.readlines()
#print(len(lines))
