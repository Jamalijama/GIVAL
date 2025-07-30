import pandas as pd
import subprocess
import sys

def run_cmd_map():
    cmd_command = 'python map_after_blast.py'
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()


def run_cmd_IAV_predict():
    cmd_command = 'python IAV_predicting.py'
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_IAV_predict_analysis():
    cmd_command = 'python IAV_predicting_analysis.py'
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_IAV_predict_whole():
    cmd_command = 'python IAV_predicting_whole.py'
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_IAV_predict_test():
    cmd_command = 'python IAV_predicting_for_test_final.py'
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_CoV_cut():
    cmd_command = 'python CoV_cutting.py'
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_CoV_cut_whole():
    cmd_command = 'python CoV_cutting_whole.py'
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_other_family_cut():
    cmd_command = 'python other_family_cutting.py'
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_other_family_cut_whole():
    cmd_command = 'python other_family_cutting_whole.py'
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()


def run_cmd_CoV_predict():
    cmd_command = 'python CoV_predicting.py'
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_other_family_predict():
    cmd_command = 'python other_family_predicting.py'
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_CoV_and_other_predict_analysis():
    cmd_command = 'python CoV_and other_predicting_analysis.py'
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_extract_feature_CoV_test():
    cmd_command = 'python try.py'
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_extract_feature_CoV_S_all():
    cmd_command = '''
cd ..
python extract_features.py \
  --input_file=./GIVAL/txt_npy_file/new/for_predict_model_token_onlywith_HMM_S.txt \
  --output_file=./GIVAL/txt_npy_file/new/for_predict_model_token_onlywith_HMM_S.jsonl \
  --vocab_file=./vBERT_optimized/vocab.txt \
  --do_lower_case=False \
  --bert_config_file=./vBERT_optimized/bert_config.json \
  --init_checkpoint=./vBERT_optimized/model/model.ckpt-380000 \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=256 \
  --batch_size=8
'''

    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()


def run_cmd_extract_feature_other_family_all():
    cmd_command = '''
cd ..
python extract_features.py \
  --input_file=./GIVAL/txt_npy_file/new/for_predict_model_token_onlywith_HMM.txt \
  --output_file=./GIVAL/txt_npy_file/new/for_predict_model_token_onlywith_HMM.jsonl \
  --vocab_file=./vBERT_optimized/vocab.txt \
  --do_lower_case=False \
  --bert_config_file=./vBERT_optimized/bert_config.json \
  --init_checkpoint=./vBERT_optimized/model/model.ckpt-380000 \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=256 \
  --batch_size=8
'''

    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_jsonl2npy_CoV_S_all():
    cmd_command = '''
cd ..
python jsonl2npy_S_all_HMM.py 
'''
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_jsonl2npy_other_family_all():
    cmd_command = '''
cd ..
python jsonl2npy_other_family_all_HMM.py 
'''
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_extract_feature_CoV_S_input():
    cmd_command = '''
cd ..
python extract_features.py \
  --input_file=./GIVAL/txt_npy_file/new/token_onlywith_HMM_target_seq.txt \
  --output_file=./GIVAL/txt_npy_file/new/token_onlywith_HMM_target_seq.jsonl \
  --vocab_file=./vBERT_optimized/vocab.txt \
  --do_lower_case=False \
  --bert_config_file=./vBERT_optimized/bert_config.json \
  --init_checkpoint=./vBERT_optimized/model/model.ckpt-380000 \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=256 \
  --batch_size=8
'''

    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_jsonl2npy_CoV_S_input():
    cmd_command = '''
cd ..
python jsonl2npy_S_input_HMM.py 
'''
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_extract_feature_other_family_input():
    cmd_command = '''
cd ..
python extract_features.py \
  --input_file=./GIVAL/txt_npy_file/new/token_onlywith_HMM_target_seq.txt \
  --output_file=./GIVAL/txt_npy_file/new/token_onlywith_HMM_target_seq.jsonl \
  --vocab_file=./vBERT_optimized/vocab.txt \
  --do_lower_case=False \
  --bert_config_file=./vBERT_optimized/bert_config.json \
  --init_checkpoint=./vBERT_optimized/model/model.ckpt-380000 \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=256 \
  --batch_size=8
'''

    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_jsonl2npy_other_family_input():
    cmd_command = '''
cd ..
python jsonl2npy_other_family_input_HMM.py 
'''
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()


def run_cmd_extract_feature_IAV_input():
    cmd_command = '''
cd ..
python extract_features.py \
  --input_file=./GIVAL/txt_npy_file/new/token_onlywith_HMM.txt \
  --output_file=./GIVAL/txt_npy_file/new/token_onlywith_HMM.jsonl \
  --vocab_file=./vBERT_optimized/vocab.txt \
  --do_lower_case=False \
  --bert_config_file=./vBERT_optimized/bert_config.json \
  --init_checkpoint=./vBERT_optimized/model/model.ckpt-380000 \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=256 \
  --batch_size=8
'''

    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def run_cmd_jsonl2npy_IAV_input():
    cmd_command = '''
cd ..
python jsonl2npy_IAV_input_HMM.py 
'''
    process = subprocess.Popen(cmd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

run_cmd_map()
print('run_cmd_map_finished!')
aa = str(sys.argv[1])
df_loc = pd.read_csv('./result/df_loc.csv')
df_loc_lst = list(df_loc['loc'])
gene = str(df_loc_lst[2])
family = str(df_loc_lst[-1])
print(family)
print(gene)
if family == 'CoV':
    if aa == 'segment':
        run_cmd_CoV_cut()
        print('run_cmd_CoV_cut_finished!')
    elif aa == 'complete':
        run_cmd_CoV_cut_whole()
        print('run_cmd_CoV_cut_whole_finished!')
    run_cmd_extract_feature_CoV_S_all()
    print('run_cmd_extract_feature_CoV_S_all_finished!')
    run_cmd_jsonl2npy_CoV_S_all()
    print('run_cmd_jsonl2npy_CoV_S_all_finished!')
    run_cmd_extract_feature_CoV_S_input()
    print('run_cmd_extract_feature_CoV_S_input_finished!')
    run_cmd_jsonl2npy_CoV_S_input()
    print('run_cmd_jsonl2npy_CoV_S_input_finished!')
    run_cmd_CoV_predict()
    print('run_cmd_CoV_predict_finished!')
    run_cmd_CoV_and_other_predict_analysis()
    print('run_cmd_CoV_and_other_predict_analysis_finished!')
elif family == 'IVA':
    run_cmd_extract_feature_IAV_input()
    print('run_cmd_extract_feature_IAV_input_finished!')
    run_cmd_jsonl2npy_IAV_input()
    print('run_cmd_jsonl2npy_IAV_input_finished!')
    if aa == 'segment':
        run_cmd_IAV_predict()
        print('run_cmd_IAV_predict_finished!')
          
        #run_cmd_IAV_predict_test()
        #print('run_cmd_IAV_predict_test_finished!')

    elif aa == 'complete':
        run_cmd_IAV_predict_whole()
        print('run_cmd_IAV_predict_whole_finished!')
    run_cmd_IAV_predict_analysis() 
    print('run_cmd_IAV_predict_analysis_finished!')

else:
    if aa == 'segment':
        run_cmd_other_family_cut()
        print('run_cmd_other_family_cut_finished!')
    elif aa == 'complete':
        run_cmd_other_family_cut_whole()
        print('run_cmd_other_family_cut_whole_finished!')
    run_cmd_extract_feature_other_family_all()
    print('run_cmd_extract_feature_other_family_all_finished!')
    run_cmd_jsonl2npy_other_family_all()
    print('run_cmd_jsonl2npy_other_family_all_finished!')
    run_cmd_extract_feature_other_family_input()
    print('run_cmd_extract_feature_other_family_input_finished!')
    run_cmd_jsonl2npy_other_family_input()
    print('run_cmd_jsonl2npy_other_family_input_finished!')
    run_cmd_other_family_predict()
    print('run_cmd_other_family_predict_finished!')
    run_cmd_CoV_and_other_predict_analysis()
    print('run_cmd_CoV_and_other_predict_analysis_finished!')
