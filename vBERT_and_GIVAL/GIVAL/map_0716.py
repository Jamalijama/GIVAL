import numpy as np
import pandas as pd
import jieba4.jieba as jieba
import sys


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
df_other_family = pd.concat([df_other_family0,df_other_family1],ignore_index=True)
other_ref_seq = df_other_family['seq'].tolist()
other_gene = df_other_family['gene'].tolist()
other_family = df_other_family['family'].tolist()

family_lst = ['IVA','IVA','IVA','IVA','IVA','IVA','IVA','IVA','CoV','CoV']
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
ref_seq_lst = ref_seq_lst+ ref_seq_wuhan
lst_family.append('CoV')
lst_segment.append('S')
ref_seq_lst += other_ref_seq
lst_family += other_family
lst_segment += other_gene
#target_seq = 'MVMELIRMVKRGINDRNFWRGENGRKTRSAYERMCNILKGKFQTAAQRAMVDQVRESRNPGNAEIEDLIFLARSALILRGSVAHKSCLPACAYGPAVSSGYDFEKEGYSLVGIDPFKLLQNSQIYSLIRPNENPAHKSQLVWMACHSA'
#target_seq = 'GACPRYVKQSTLKLATGMRNVPEKQTRGIFGAIAGFIENGWEGMVDGWYGFRHQNSEGRGQAADLKSTQAAIDQVNGKLNRLIGKTNEKFHQIEKEFSEVEGRVQDLEKYVEDTKIDLWSYNAELLVALENQHTIDLTDSEMNKLFEKTKKQLRENAEDMGNGCFKIYHKCDNACIGSIRNETYDHNVYRDEALNNRFQIKGVELKSGYKDWILWISFAMSCFLLCIALLGFIMWA'
#target_seq = 'GACPRYVKQSTLKLATGMRNVPEKQTRGIFGAIAGFIENGWEGMVDGWYGFRHQNSEGRGQAADLKSTQAAIDQVNGKLNRLIGKTNEKFHQIEKEFSEVEGRVQDLEKYVEDTKIDLWSYNAELLVALENQHTIDLTDSEMNKLFEKTKKQLRENAEDMGNG'

#target_seq='RNLYKPENLTFANVIAVSRGANYTTLNKTFDIPELNSTFPIEEEFREYFQNMSSELQVLKNLTADMSKLNISAEIQLINEIAHNVSNMRVEVEKFQRYVNYVKWAWWQ'
#MKMASNDATVAVACNNNNDKEKSSGEGLFTNMSSTLKKALGARPKQPAPRDKPQKPPRPPTPELVKRIPPPPPNGEEEEEPVIRYEVKSGISGLPELTTVPQPDVANTAFSVPPLSLRENK
#FGFCTASEAVSYYSEAAASGFVQCRFVSFDLADTVEGLLPEDYVMVVVGTTKLSAYVDTFGSRPRNICGWLLFSNCNYFLEELELTFGRRG(ORF1ab)
#ATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTG(spike_RBD)
#DTLCIGYHANNSTDTVDTVLEKNVTVTHSVNLLEDKHNGKLCKLRGVAPLHLGKCNIAGWILGNPECESLSTASSWSYIVETPSSDNGTCYPGDFIDYEELREQLSSVSSFERFEIFPKTSSWPNHDSNKGVTAACPHAGAKSFYKNLIWLVKKGNSYPKLSKSYINDKGKEVLVLWGIHHPSTSADQQSLYQNADAYVFVGSSRYSKKFKPEIAIRPKVR~~EGRMNYYWTLVEPGDKITFEATGNLVVPRYAFAMERNAGSGIIISDTPVHDCNTTCQTPKGAINTSLPFQNIHPITIGKCPKYVKSTKLRLATGLRNIPSIQSR(HA1)
#SPANDLCYPGDFNDYEELKHLLSRTNHFEKIQIIPKSSWSNHDASSGVSSACPYHGRSSFFRNVVWLIKKNSAYPTIKRSYNNTNQEDLLVLWGIHHPNDAAEQTKLYQNPTTYISVGTSTLNQRLVPEIATRPKVNGQSGRMEFFWTILK(HA_RBD_100-250from1_h5n1)
target_seq = str(sys.argv[1])
shortcut = 20

ref_lev_res = [[] for _ in range(len(ref_seq_lst))]
ref_dis_lst = []
for i, ref in enumerate(ref_seq_lst):
    print(i)
    start_cut = target_seq[:shortcut]
    end_cut = target_seq[-shortcut:]
    # print(len(start_cut), len(end_cut))
    start_lev_lst = []
    end_lev_lst = []
    for j in range(0, len(ref) - shortcut, 1):
        start_lev_lst.append(lev_distance(start_cut, ref[j:j + shortcut]))
        end_lev_lst.append(lev_distance(end_cut, ref[j:j + shortcut]))
    start = start_lev_lst.index(min(start_lev_lst))
    end = end_lev_lst.index(min(end_lev_lst))

    # print(len(target_seq), len(ref[start:end + shortcut]))
    ref_lev_dis = lev_distance(target_seq, ref[start:end + shortcut])
    ref_lev_res[i].append(start)
    ref_lev_res[i].append(end + shortcut)
    ref_lev_res[i].append(ref_lev_dis)
    ref_dis_lst.append(ref_lev_dis)
df_map = pd.DataFrame()
df_map['ref_seq'] = ref_seq_lst
df_map['family'] = lst_family
df_map['segment'] = lst_segment
df_map['LD'] = ref_dis_lst
df_map.to_csv('df_map.csv',index=False)

print(ref_lev_res)
array_ref_lev = np.array(ref_lev_res)
print(array_ref_lev.shape)

# search for best reference
# print(array_ref_lev[:, 2])
opt_index = np.argmin(array_ref_lev[:, 2])
print(opt_index)

loc_lst = ref_lev_res[opt_index][:2]

loc_lst.append(lst_segment[opt_index])
start_loc = ref_lev_res[opt_index][0]
end_loc = ref_lev_res[opt_index][1]
loc_lst.append(ref_seq_lst[opt_index][start_loc:end_loc])
loc_lst.append(lst_family[opt_index])
df = pd.DataFrame()
df['loc'] = loc_lst
#df.to_csv('./result/df_' + lst_segment[opt_index] + '_0523.csv', index=False)
df.to_csv('./result/df_loc.csv', index=False)

target_seq_new = ref_seq_lst[opt_index][:start_loc] + target_seq + ref_seq_lst[opt_index][end_loc:]
# print(len(ref_seq_lst[indexx]))
# print(len(target_seq_new))

amino_num = 4
gene = lst_segment[opt_index]
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
