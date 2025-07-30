import numpy as np
import pandas as pd
import sys
import time
import jieba4.jieba as jieba

query_seq = 'SPANDLCYPGDFNDYEELKHLLSRTNHFEKIQIIPKSSWSNHDASSGVSSACPYHGRSSFFRNVVWLIKKNSAYPTIKRSYNNTNQEDLLVLWGIHHPNDAAEQTKLYQNPTTYISVGTSTLNQRLVPEIATRPKVNGQSGRMEFFWTILK'

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

df_other_family0 = pd.read_csv('./csv_file/ref_all_family.csv')
df_other_family1 = pd.read_csv('./csv_file/df_ref_mpox.csv')
df_other_family2 = pd.read_csv('./csv_file/ref_seq_ncbi_virus.csv')
df_other_family = pd.concat([df_other_family0,df_other_family1,df_other_family2],ignore_index=True)
other_ref_seq = df_other_family['seq'].tolist()
other_gene = df_other_family['gene'].tolist()
other_family = df_other_family['family'].tolist()

family_lst = ['IAV','IAV','IAV','IAV','IAV','IAV','IAV','IAV','CoV','CoV']
seg_lst = ['PB2','PB1','PA','NP', 'HA', 'NA','M','NS','S','ORF1ab']
cds_lst = ['1','2','3','5', '4', '6','7','8', 'S','ORF1ab']
lst_family = []
lst_segment = []
ref_seq_lst = []
for i in range(len(seg_lst)):
    if i == 8:
        file = './csv_file/S_without_test_set_DCR_ref.csv'
        df = pd.read_csv(file)
        for j in range(df.shape[0]):
            lst_segment.append(seg_lst[i])
            lst_family.append(family_lst[i])
        ref_seq_lst += df['CDS_' + cds_lst[i] + '_amino'].tolist()

    elif i == 9:
        file = './csv_file/S_without_test_set_DCR_ref.csv'
        df = pd.read_csv(file)
        for j in range(df.shape[0]):
            lst_segment.append(seg_lst[i])
            lst_family.append(family_lst[i])
        ref_seq_lst += df['CDS_' + cds_lst[i] + '_amino'].tolist()

    elif (i == 3)|(i == 4)|(i == 5):
        file = './csv_file/' + seg_lst[i] + '_without_test_set_DCR_human_avian_ref.csv'

        df = pd.read_csv(file)
        for j in range(df.shape[0]):
            lst_segment.append(seg_lst[i])
            lst_family.append(family_lst[i])
        ref_seq_lst += df['CDS_' + cds_lst[i] + '_amino_seq'].tolist()
    else:
        file = './csv_file/IVA_other_gene_ref.csv'
        df = pd.read_csv(file)
        for j in range(df.shape[0]):
            lst_segment.append(seg_lst[i])
            lst_family.append(family_lst[i])
        ref_seq_lst += df['CDS_' + cds_lst[i] + '_amino_seq'].tolist()
#ref_seq_lst.append('MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHA~~~SGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLG~~YHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIDDTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVSIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSRRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILARLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTHNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT')
df_csv111 = pd.read_csv('./csv_file/ref_WuHan_id_seq_with_RBD.csv')
ref_seq_wuhan = df_csv111['amino_seq'].tolist()
#ref_seq_lst.append('MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT')
ref_seq_lst = ref_seq_lst + ref_seq_wuhan
lst_family.append('CoV')
lst_segment.append('S')
ref_seq_lst += other_ref_seq
lst_family += other_family
lst_segment += other_gene

lst_virus_gene = []
for i in range(len(lst_segment)):
    virus = lst_family[i]
    gene = lst_segment[i]
    virus_gene = virus+'_'+gene
    lst_virus_gene.append(virus_gene)

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


df_ref_all = pd.DataFrame()
df_ref_all['ref_seq'] = ref_seq_lst
df_ref_all['virus'] = lst_family
df_ref_all['gene'] = lst_segment
df_ref_all['virus_gene'] = lst_virus_gene
df_ref_all['ref_seq_with_space'] = ref_seq_lst0
df_ref_all['flag_space'] = flag_remove_seq_lst
df_ref_all['dict_loc_spaceloc'] = dict_loc_lst

df_ref_all.to_csv('./mapping_ref_seq/ref_seq_all.csv',index=False)
# df_11 = df_ref_all[df_ref_all['virus']!='monkeypox']
# df_1 = df_11.drop_duplicates(subset='virus_gene',keep='first')
# print(len(df_1))
# df_ref_all = pd.read_csv('./mapping_ref_seq/ref_seq_all.csv')
f1 = open('./mapping_ref_seq/ref_seq_all.fasta','w')
for i in range(len(df_ref_all)):
    seq_ref = str(df_ref_all.loc[i,'ref_seq']).upper()
    virus_name = str(df_ref_all.loc[i,'virus']).replace(' ','|')
    gene_name = str(df_ref_all.loc[i,'gene']).replace(' ','|')
    seq_id = virus_name+'_'+gene_name+'_'+str(i)
    print('>'+seq_id,file=f1)
    print(seq_ref,file=f1)
f1.close()
f2 = open('./mapping_ref_seq/query.fasta','w')
print('>query_seq',file=f2)
print(query_seq,file=f2)