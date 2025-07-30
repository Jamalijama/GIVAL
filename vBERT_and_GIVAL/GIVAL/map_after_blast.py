import pandas as pd
import numpy as np
import jieba4.jieba as jieba

def lev_distance(seq0, seq1):
    '''
    Parameters
    ----------
    seq0 : String
        DESCRIPTION:
            One of the two sequences to calculate Levenshtein distance.
    seq1 : String
        DESCRIPTION:
            The other of the two sequences to calculate Levenshtein distance.

    Returns
    -------
    TYPE: int.
        DESCRIPTION:
            A Levenshtein distance of int type between two sequences.
    '''

    if len(seq0) >= len(seq1):
        seqL, seqS, seqlen = seq0, seq1, len(seq0)
    else:
        seqL, seqS, seqlen = seq1, seq0, len(seq1)

    count_arr = np.arange(seqlen + 1)

    for i in range(1, len(seqL) + 1):
        ref1 = count_arr[0]
        count_arr[0] += 1

        for j in range(1, len(seqS) + 1):
            ref2 = count_arr[j]

            if seqL[i - 1] == seqS[j - 1]:
                count_arr[j] = ref1
            else:
                count_arr[j] = min(ref1, min(count_arr[j - 1], count_arr[j])) + 1
            ref1 = ref2

    return count_arr[len(seqS)]

def parse_fasta(fasta_file):
    fr = open(fasta_file, 'r')
    contents = fr.readlines()
    idlst, seqlst, seq, seqnum = [], [], '', 0

    for line in contents:
        if line[:1] != '>':
            seq = seq + line[:-1]
        else:
            seqnum += 1
            seqlst.append(seq)
            seq = ''
            seqid = line[1:-1]
            idlst.append(seqid)
    seqlst = seqlst[1:]

    seqlast, line_num_all = '', len(contents)

    for j in range(line_num_all - 1, 0, -1):
        line = contents[j]
        if line[:1] != '>':
            seqlast = line[:-1] + seqlast
        else:
            break

    seqlst.append(seqlast)
    return idlst, seqlst


fasta_file0 = './mapping_ref_seq/query.fasta'
file0 = parse_fasta(fasta_file0)
id_lst, seq_lst0 = file0[0], file0[1]
query_seq = seq_lst0[0]

col_name_lst = ['Query_id','Subject_id','Identity','Align_length','Miss_match','Gap','Query_start','Query_end','Subject_start','Subject_end','E_value','Score']

df_ = pd.read_csv('./mapping_ref_seq/seq_query.tsv',sep='\t',names=col_name_lst) # move BLAST result 'seqtsv' to 'mapping_ref_seq' folder

df_ref = pd.read_csv('./mapping_ref_seq/ref_seq_all.csv')
ref_seq_lst = list(df_ref['ref_seq_with_space'])
ref_seq_lst0 = ref_seq_lst.copy()
ref_seq_lst = []
flag_remove_seq_lst = []
dict_loc_lst = []
for seq in ref_seq_lst0:
    seq1 = seq.replace('~','')
    ref_seq_lst.append(seq1)
    if len(seq1) != len(seq):
        flag_remove_seq_lst.append(0)
        non_space_site_lst = []
        for i in range(len(seq)):
            if seq[i] != '~':
                non_space_site_lst.append(i)
        dict_loc = dict(zip(range(len(non_space_site_lst)),non_space_site_lst))
        dict_loc_lst.append(dict_loc)

    else:
        flag_remove_seq_lst.append(1)
        dict_loc_lst.append({})
df_ref['dict_loc_spaceloc'] = dict_loc_lst

mapped_seq_id = df_.loc[0,'Subject_id']
mapped_seq_id_lst = mapped_seq_id.split('_')
mapped_virus = mapped_seq_id_lst[0]
mapped_gene = mapped_seq_id_lst[1]
mapped_id = int(mapped_seq_id_lst[2])

mapped_seq = df_ref.loc[mapped_id,'ref_seq_with_space']
mapped_start_loc0 = int(df_.loc[0,'Subject_start'])-1
mapped_end_loc0_minus1 = int(df_.loc[0,'Subject_end'])-1

mapped_seq0 = df_ref.loc[mapped_id,'ref_seq']
mapped_seg0_len = mapped_end_loc0_minus1-mapped_start_loc0+1
if mapped_seg0_len > len(query_seq):
    m_s = mapped_seq0[mapped_start_loc0:mapped_end_loc0_minus1+1]
    m_s_1 = mapped_seq0[mapped_start_loc0:mapped_end_loc0_minus1]
    if lev_distance(query_seq,m_s_1)<lev_distance(query_seq,m_s):
        mapped_end_loc0_minus1 = mapped_end_loc0_minus1-1

if int(df_ref.loc[mapped_id,'flag_space']) == 1:
    mapped_start_loc = mapped_start_loc0
    mapped_end_loc = mapped_end_loc0_minus1+1
else:
    dict_seq = df_ref.loc[mapped_id,'dict_loc_spaceloc']
    mapped_start_loc = int(dict_seq[mapped_start_loc0])
    mapped_end_loc = int(dict_seq[mapped_end_loc0_minus1])+1
mapped_ref_seg = mapped_seq[mapped_start_loc:mapped_end_loc]
print(mapped_ref_seg)

df_loc = pd.DataFrame()
loc_lst = []
if mapped_virus == 'IAV':
    mapped_virus = 'IVA'
loc_lst = [mapped_start_loc,mapped_end_loc,mapped_gene,mapped_ref_seg,mapped_virus]
df_loc['loc'] = loc_lst
df_loc.to_csv('./result/df_loc.csv',index=False)

target_seq = query_seq
target_seq_new = mapped_seq[:mapped_start_loc] + target_seq + mapped_seq[mapped_end_loc:]

amino_num = 4
gene = mapped_gene
print(gene)
#fw1 = open('./txt_npy_file/new/token_' + str(amino_num) + 'aa_' + gene + '_0523.txt', 'w')
fw1 = open('./txt_npy_file/new/token_' + str(amino_num) + 'aa.txt', 'w')

amino_seq_new = target_seq_new
sl = []
for b in range(0,len(amino_seq_new),amino_num):
    amino_site = amino_seq_new[b:b+amino_num]
    sl.append(amino_site)

CDS_token_sentence_new = ' '.join(sl)
print(CDS_token_sentence_new, file=fw1)

fw1.close()

# 切换词库
jieba.set_dictionary("dict_empty.txt")

#fw2 = open('./txt_npy_file/token_onlywith_HMM_' + gene + '_0523.txt', 'w')
fw2 = open('./txt_npy_file/new/token_onlywith_HMM.txt', 'w')

sl = jieba.lcut(amino_seq_new,HMM=True)
CDS_token_sentence_new = ' '.join(sl)
print(CDS_token_sentence_new, file=fw2)

fw2.close()

fw3 = open('./txt_npy_file/new/token_' + str(amino_num) + 'aa_target_seq.txt', 'w')

amino_seq_new = target_seq
sl = []
for b in range(0,len(amino_seq_new),amino_num):
    amino_site = amino_seq_new[b:b+amino_num]
    sl.append(amino_site)

CDS_token_sentence_new = ' '.join(sl)
print(CDS_token_sentence_new, file=fw3)

fw3.close()

# 切换词库
jieba.set_dictionary("dict_empty.txt")

#fw2 = open('./txt_npy_file/token_onlywith_HMM_' + gene + '_0523.txt', 'w')
fw4 = open('./txt_npy_file/new/token_onlywith_HMM_target_seq.txt', 'w')

sl = jieba.lcut(amino_seq_new,HMM=True)
CDS_token_sentence_new = ' '.join(sl)
print(CDS_token_sentence_new, file=fw4)

fw4.close()


mapped_seq_lst = [mapped_seq.replace('~',''),mapped_seq]
df_mapped_seq = pd.DataFrame()
df_mapped_seq['mapped_ref'] = mapped_seq_lst
df_mapped_seq.to_csv('./result/mapped_ref_seq.csv',index=False)
