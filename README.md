# GIVAL
General Intelligence to predict Virus Adaptation based on genome Language model

Data and models need to be downloaded from Zenodo (https://10.5281/zenodo.14233092) and placed to the corresponding folders before run the scripts in GitHub (https://github.com/Jamalijama/GIVAL). Scripts on GitHub only included codes related to GIVAL framework and other script packages also need to be downloaded from Zenodo.

Note: In the dataset or program files, 'country' represents country or area.

This package provides an pipeline of data parsing, tokenization and segmentation of gene sequences, pretraining and embedding evaluation of vBERT and other language models, analysis of immune escape and receptor binding based on the vBERT embedding, establishment of GIVAl based on vBERT and predicting of IVA HA RBD and CoV Spike RBD sequences.

## Creating embedding npy file for the following steps
Please extract embedding with the following commands before other steps.
    ```bash
    python vBERT_and_GIVAL/extract_features.py \
  --input_file=./input.txt \
  --output_file=./input.jsonl \
  --vocab_file=./vBERT_optimized/vocab.txt \
  --do_lower_case=False \
  --bert_config_file=./vBERT_optimized/bert_config.json \
  --init_checkpoint=./vBERT_optimized/model/model.ckpt-380000 \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=256 \
  --batch_size=8
     ```
In the comannds above, 'input' should be replaced with the path and input txt file name.
The 'dirr' and 'file_lst' in python file `vBERT_and_GIVAL/jsonl2npy.py` should be replaced with the path and input txt file name, respectively, and the following commands are ultilized to create the npy embedding file.
    ```bash
    python vBERT_and_GIVAL/jsonl2npy.py \
     ```
Following the above commands, the npy files of embedding of the following txt files need to be created.
`vBERT_and_GIVAL/vBERT_optimized/test_sample/try_AIV_sample_token_onlywith_HMM_CDS_5_NP_human_avian1000_38w.txt`, `vBERT_and_GIVAL/embedding_evaluation/HA_testset_for_embedding_evaluation/try_AIV_sample_token_onlywith_HMM_CDS_4_HA_serotype2000_38w.txt`, `vBERT_and_GIVAL/embedding_evaluation/Spike_RBD_testset_for_embedding_evaluation/try_Spike_sample_token_onlywith_HMM_seg_RBD_variant1000_38w_256cut_HMM.txt`, every txt file in `vBERT_and_GIVAL/GIVAL/txt_npy_file`, `vBERT_and_GIVAL/GIVAL/test_set/token_onlywith_HMM_HA_test_set.txt`, `vBERT_and_GIVAL/GIVAL/test_set/test_canine/token_onlywith_HMM_other_host.txt`, `immune_escape_and_binding_analysis/DMS_dataset_analysis/Spike_RBD_binding/RBD_mutants_38w_HMM.txt`, `immune_escape_and_binding_analysis/DMS_dataset_analysis/HA_preference/HA_H1_mutants_38w_HMM.txt`, `immune_escape_and_binding_analysis/IVA_HA_vaccine_evaluate/HA_H1N1_after2021_with_WHOref_38w_HMM.txt`, `immune_escape_and_binding_analysis/IVA_HA_vaccine_evaluate/HA_H3N2_after2021_with_WHOref_38w_HMM.txt`.

## Data parsing

The datasets for HMM tokenizer training and language model pretraining are in `data/file`.

1.  File `data/file/df_all_after_deduplication.csv` is the whole dataset after sequence deduplication.

2.  File `data/file/df_all_after_sampled_to_20w.csv` is the sampled dataset for HMM tokenizer training (please run the py files in `data` to conduct sequence sampling).

3.  File `data/file/df_all_after_sampled_to_10w.csv` is the sampled dataset for vBERT pretraining.

4.  File `data/file/df_all_after_sampled_to_10w.csv` and `data/file/df_10000sample_5time_HA.csv` is the simulated dataset.

5. Run `data/sampled_data_evaluation/sample_data_evaluate.py` and `data/sampled_data_evaluation/KW_test.py` to evaluate the sampled dataset.

    ```bash
    python data/sampled_data_evaluation/sample_data_evaluate.py
    python data/sampled_data_evaluation/KW_test.py
    ```

## Sequence tokenization and segmentation

1.  Run `tokenizing_and_segmentating/1fixed_aa_word_count.py`, `tokenizing_and_segmentating/2dict_to_txt.py`, `tokenizing_and_segmentating/3jieba_delete_useless_token.py`, `tokenizing_and_segmentating/4dict_to_txt_final.py`,  `tokenizing_and_segmentating/5token_without_HMM.py`, `tokenizing_and_segmentating/6_1count_freq_prob_emit.py`, `tokenizing_and_segmentating/6_2count_freq_prob_start.py` and  `tokenizing_and_segmentating/6_3count_freq_prob_trans.py` to train a HMM tokenizer for virus protein sequences.

    ```bash
    python tokenizing_and_segmentating/1fixed_aa_word_count.py
    python tokenizing_and_segmentating/2dict_to_txt.py
    python tokenizing_and_segmentating/3jieba_delete_useless_token.py
    python tokenizing_and_segmentating/4dict_to_txt_final.py
    python tokenizing_and_segmentating/5token_without_HMM.py
    python tokenizing_and_segmentating/6_1count_freq_prob_emit.py
    python tokenizing_and_segmentating/6_2count_freq_prob_start.py
    python tokenizing_and_segmentating/6_3count_freq_prob_trans.py
    ```

2.  Run `tokenizing_and_segmentating/7_1token_simulated_data_HMM.py`, `tokenizing_and_segmentating/7_2token_fixed_aa_sampled_to_10w.py`, `tokenizing_and_segmentating/7_3token_sampled_to_10w_HMM.py` and `tokenizing_and_segmentating/7_4token_whole_datasetafter_deduplication_HMM.py` to conduct sequence tokenization of simulated dataset,sampled dataset and whole dataset after deduplication based on fixed length of amino acids and HMM.

    ```bash
    python tokenizing_and_segmentating/7_1token_simulated_data_HMM.py
    python tokenizing_and_segmentating/7_2token_fixed_aa_sampled_to_10w.py
    python tokenizing_and_segmentating/7_3token_sampled_to_10w_HMM.py
    python tokenizing_and_segmentating/7_4token_whole_datasetafter_deduplication_HMM.py
    ```

3. Run `tokenizing_and_segmentating/8segmentating.py` for tokenized sequences segmentation.

    ```bash
    python tokenizing_and_segmentating/8segmentating.py
    ```
4. Run `tokenizing_and_segmentating/9create_vocabulary_list_for_pretraining.py` for creating the vocabulary list for pretraining of language models.

    ```bash
    python tokenizing_and_segmentating/9create_vocabulary_list_for_pretraining.py
    ```
5. Run `tokenizing_and_segmentating/10_1coverage_of_family_no_weight.py`,`tokenizing_and_segmentating/10_2coverage_of_family_no_weight_other_viruses.py` and `tokenizing_and_segmentating/10_3coverage_of_family_weight.py` to calculate the vocabulary coverage of each family.

    ```bash
    python tokenizing_and_segmentating/10_1coverage_of_family_no_weight.py
    python tokenizing_and_segmentating/10_2coverage_of_family_no_weight_other_viruses.py
    python tokenizing_and_segmentating/10_3coverage_of_family_weight.py
    ```
6. Run `tokenizing_and_segmentating/virtual_seq/create_virtual_seq_20w_1to1.py` to create virtual sequences based on the HMM dataset. 

    ```bash
    python tokenizing_and_segmentating/virtual_seq/create_virtual_seq_20w_1to1.py
    ```

The parameters of HMM tokenizer of virtual dataset was also calculated with the same method with step 1.

7. Run `tokenizing_and_segmentating/11family_vec.py`, `tokenizing_and_segmentating/12_1count_family_num_for_more_than500_DNA_cos.py`, `tokenizing_and_segmentating/12_2count_family_num_for_more_than500_RNA_cos.py`,`tokenizing_and_segmentating/13_1count_vocab_with_high_freq_DNA.py` and `tokenizing_and_segmentating/13_2count_vocab_with_high_freq_RNA.py` for further analysis of HMM tokenizer based on the frequency of tokens and relationship between families.

    ```bash
    python tokenizing_and_segmentating/11family_vec.py
    python tokenizing_and_segmentating/12_1count_family_num_for_more_than500_DNA_cos.py
    python tokenizing_and_segmentating/12_2count_family_num_for_more_than500_RNA_cos.py
    python tokenizing_and_segmentating/13_1count_vocab_with_high_freq_DNA.py
    python tokenizing_and_segmentating/13_2count_vocab_with_high_freq_RNA.py
    ```

## Pretraining and embedding evaluation of vBERT and other language models

1. The vBERT models were pretrained with the original BERT framework based on different datasets, tokenization and segmentation methods, parameters. The pretrained vBERT-optimized model is in `vBERT_and_GIVAL/vBERT_optimized/model`, and `vBERT_and_GIVAL/vBERT_optimized/sample.txt` and `vBERT_and_GIVAL/vBERT_optimized/vocab.txt` are the pretraining dataset and vocabulary list. Other vBERT models for parameter optimization are in `vBERT_and_GIVAL/other_models_for_parameter_optimization`. Run `vBERT_and_GIVAL/embedding_evaluation/HA_testset_for_embedding_evaluation/embedding_padding.py`, `vBERT_and_GIVAL/embedding_evaluation/HA_testset_for_embedding_evaluation/embedding_evaluation_tSNE.py`, `vBERT_and_GIVAL/embedding_evaluation/Spike_RBD_testset_for_embedding_evaluation/embedding_padding_and_evaluation_tSNE.py`, `vBERT_and_GIVAL/vBERT_optimized/test_sample/0_1_loc_clustering_final.py` and `vBERT_and_GIVAL/vBERT_optimized/test_sample/0_2cluster_with_all_site_tokens.py` for embedding evaluation of vBERT models.

    ```bash
    python vBERT_and_GIVAL/embedding_evaluation/HA_testset_for_embedding_evaluation/embedding_padding.py
    python vBERT_and_GIVAL/embedding_evaluation/HA_testset_for_embedding_evaluation/embedding_evaluation_tSNE.py
    python vBERT_and_GIVAL/embedding_evaluation/Spike_RBD_testset_for_embedding_evaluation/embedding_padding_and_evaluation_tSNE.py
    python vBERT_and_GIVAL/vBERT_optimized/test_sample/0_1_loc_clustering_final.py
    python vBERT_and_GIVAL/vBERT_optimized/test_sample/0_2cluster_with_all_site_tokens.py
    ```

2. Run `vBERT_and_GIVAL/Transformer/transformer_768_2e-4_16.py` for Transformer model pretraining. Pretrained Transformer model is in `vBERT_and_GIVAL/Transformer/model_768_2e-4_16`. Run `vBERT_and_GIVAL/Transformer/embedding_eval_transformer_HA_768_2e-4_2048.py` and `vBERT_and_GIVAL/Transformer/embedding_eval_transformer_RBD_768_2e-4_2048.py` for embedding evaluation of Transformer model.

    ```bash
    python vBERT_and_GIVAL/Transformer/transformer_768_2e-4_16.py
    python vBERT_and_GIVAL/Transformer/embedding_eval_transformer_HA_768_2e-4_2048.py
    python vBERT_and_GIVAL/Transformer/embedding_eval_transformer_RBD_768_2e-4_2048.py
    ```
3. Run `vBERT_and_GIVAL/Word2Vec/test_sample/word2vec_NP.py`,  `vBERT_and_GIVAL/Word2Vec/test_sample/0_1_loc_clustering_final.py` and `vBERT_and_GIVAL/Word2Vec/test_sample/0_2cluster_with_all_site_tokens.py` for embedding and evaluation of Word2Vec model.

    ```bash
    python vBERT_and_GIVAL/Word2Vec/test_sample/word2vec_NP.py
    python vBERT_and_GIVAL/Word2Vec/test_sample/0_1_loc_clustering_final.py
    python vBERT_and_GIVAL/Word2Vec/test_sample/0_2cluster_with_all_site_tokens.py
    ```

## Analysis of immune escape and receptor binding based on the vBERT embedding

1. Run `immune_escape_and_binding_analysis/IVA_HA_vaccine_evaluate/padding_dimensional_reduction_clustering_HA_H1N1.py` and `immune_escape_and_binding_analysis/IVA_HA_vaccine_evaluate/padding_dimensional_reduction_clustering_HA_H3N2.py` to analyze the relationship between the WHO reference vaccine strains and circulating strains based on vBERT-optimized embedding.

    ```bash
    python immune_escape_and_binding_analysis/IVA_HA_vaccine_evaluate/padding_dimensional_reduction_clustering_HA_H1N1.py
    python immune_escape_and_binding_analysis/IVA_HA_vaccine_evaluate/padding_dimensional_reduction_clustering_HA_H3N2.py
    ```

2. Run `immune_escape_and_binding_analysis/DMS_dataset_analysis/HA_preference/padding_dimensional_reduction_clustering_HA_H1_mutants.py` and `immune_escape_and_binding_analysis/DMS_dataset_analysis/HA_preference/KW_test_PCA1_var_with_entropy_hist.py` to analyze the relationship between the vBERT embedding and amino acid preference of IVA HA in DMS dataset.

    ```bash
    python immune_escape_and_binding_analysis/DMS_dataset_analysis/HA_preference/padding_dimensional_reduction_clustering_HA_H1_mutants.py
    python immune_escape_and_binding_analysis/DMS_dataset_analysis/HA_preference/KW_test_PCA1_var_with_entropy_hist.py
    ```

3. Run `immune_escape_and_binding_analysis/DMS_dataset_analysis/Spike_RBD_binding/create_label_median.py` and `immune_escape_and_binding_analysis/DMS_dataset_analysis/Spike_RBD_binding/RBD_mutants_PCA_bind_avg_each_site.py` to analyze the relationship between the vBERT embedding and binding ability of Spike RBD in DMS dataset.

    ```bash
    python immune_escape_and_binding_analysis/DMS_dataset_analysis/Spike_RBD_binding/create_label_median.py
    python immune_escape_and_binding_analysis/DMS_dataset_analysis/Spike_RBD_binding/RBD_mutants_PCA_bind_avg_each_site.py
    ```
## Establishment of GIVAl based on vBERT

Run `vBERT_and_GIVAL/GIVAL/run_cmd.py` to input query sequence and flexibly conduct mapping, optimization of data and labels, training ResNet classifier and predicting to establish GIVAL based on the pretrained vBERT-optimized.

    ```bash
    python vBERT_and_GIVAL/GIVAL/run_cmd.py SPANDLCYPGDFNDYEELKHLLSRTNHFEKIQIIPKSSWSNHDASSGVSSACPYHGRSSFFRNVVWLIKKNSAYPTIKRSYNNTNQEDLLVLWGIHHPNDAAEQTKLYQNPTTYISVGTSTLNQRLVPEIATRPKVNGQSGRMEFFWTILK
    ```

NOTE: HA RBD: SPANDLCYPGDFNDYEELKHLLSRTNHFEKIQIIPKSSWSNHDASSGVSSACPYHGRSSFFRNVVWLIKKNSAYPTIKRSYNNTNQEDLLVLWGIHHPNDAAEQTKLYQNPTTYISVGTSTLNQRLVPEIATRPKVNGQSGRMEFFWTILK
Spike RBD: ATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTG

If more sequences from other viruses need to be added to the dataset of GIVAL, please add more data to `vBERT_and_GIVAL/GIVAL/csv_file/seq_all_family.csv` and reference sequences to `vBERT_and_GIVAL/GIVAL/csv_file/ref_all_family.csv`.

##Predicting based on GIVAL

### Performance evaluation of GIVAL

1. Run `predicting/performance_evaluation/mapping_score/mapping_LD_score_calculate_HA_RBD.py` and `predicting/other_host_predicting/performance_evaluation/mapping_score/mapping_LD_score_calculate_S_RBD.py` to calculate the mapping score of HA RBD and Spike RBD.

    ```bash
    python  predicting/other_host_predicting/performance_evaluation/mapping_score/mapping_LD_score_calculate_HA_RBD.py
    python  predicting/other_host_predicting/performance_evaluation/mapping_score/mapping_LD_score_calculate_S_RBD.py
    ```

2. Run `predicting/performance_evaluation/independent_validation_set/result_labeling.py` and `predicting/performance_evaluation/independent_validation_set/Confusion_Mx_Roc.py` to evaluate the performance of GIVAL on the independent validation set.

    ```bash
    python  predicting/performance_evaluation/independent_validation_set/result_labeling.py
    python  predicting/performance_evaluation/independent_validation_set/Confusion_Mx_Roc.py
    ```

3. Run `predicting/performance_evaluation/flexible_label/flexible_label_information/HA_whole_seq/heatmap_whole_HA.py`, `predicting/performance_evaluation/flexible_label/flexible_label_information/HA_whole_seq/heatmap_sampled_HA.py`, `predicting/performance_evaluation/flexible_label/flexible_label_information/HA_RBD/heatmap_whole_HA_RBD.py`, `predicting/performance_evaluation/flexible_label/flexible_label_information/HA_RBD/heatmap_sampled_HA_RBD.py`, `predicting/performance_evaluation/flexible_label/flexible_label_information/Spike_RBD/heatmap_whole_S_RBD.py` and `predicting/performance_evaluation/flexible_label/flexible_label_information/Spike_RBD/heatmap_sampled_S_RBD.py` to conduct statistical analysis of flexible labels of HA whole sequence, HA RBD and Spike RBD dataset (whole and sampled).

    ```bash
    python  predicting/performance_evaluation/flexible_label/flexible_label_information/HA_whole_seq/heatmap_whole_HA.py
    python  predicting/performance_evaluation/flexible_label/flexible_label_information/HA_whole_seq/heatmap_sampled_HA.py
    python  predicting/performance_evaluation/flexible_label/flexible_label_information/HA_RBD/heatmap_whole_HA_RBD.py
    python  predicting/performance_evaluation/flexible_label/flexible_label_information/HA_RBD/heatmap_sampled_HA_RBD.py
    python  predicting/performance_evaluation/flexible_label/flexible_label_information/Spike_RBD/heatmap_whole_S_RBD.py
    python  predicting/performance_evaluation/flexible_label/flexible_label_information/Spike_RBD/heatmap_sampled_S_RBD.py
    ```

4. Run `vBERT_and_GIVAL/GIVAL/HA_compare_cluster_final.py`, `vBERT_and_GIVAL/GIVAL/HA_compare_cluster_final_error_rate0.05.py`, `vBERT_and_GIVAL/GIVAL/HA_compare_cluster_final_error_rate0.1.py`, `vBERT_and_GIVAL/GIVAL/HA_compare_cluster_final_error_rate0.15.py`, `vBERT_and_GIVAL/GIVAL/HA_compare_cluster_final_error_rate0.2.py` and `vBERT_and_GIVAL/GIVAL/HA_compare_host_final.py`, `vBERT_and_GIVAL/GIVAL/HA_compare_host_final_0.05error_rate.py`, `vBERT_and_GIVAL/GIVAL/HA_compare_host_final_0.1error_rate.py`, `vBERT_and_GIVAL/GIVAL/HA_compare_host_final_0.15error_rate.py`, `vBERT_and_GIVAL/GIVAL/HA_compare_host_final_0.2error_rate.py` to obtain the confusion matrices under label error rate of 0, 0.05, 0.10, 0.15, 0.20. On this basis, the indexes were calculated.

    ```bash
    python  vBERT_and_GIVAL/GIVAL/HA_compare_cluster_final.py
    python  vBERT_and_GIVAL/GIVAL/HA_compare_cluster_final_error_rate0.05.py
    python  vBERT_and_GIVAL/GIVAL/HA_compare_cluster_final_error_rate0.1.py
    python  vBERT_and_GIVAL/GIVAL/HA_compare_cluster_final_error_rate0.15.py
    python  vBERT_and_GIVAL/GIVAL/HA_compare_cluster_final_error_rate0.2.py
    python  vBERT_and_GIVAL/GIVAL/HA_compare_host_final.py
    python  vBERT_and_GIVAL/GIVAL/HA_compare_host_final_0.05error_rate.py
    python  vBERT_and_GIVAL/GIVAL/HA_compare_host_final_0.1error_rate.py
    python  vBERT_and_GIVAL/GIVAL/HA_compare_host_final_0.15error_rate.py
    python  vBERT_and_GIVAL/GIVAL/HA_compare_host_final_0.2error_rate.py
    ```

5. On the basis of the calculated indexes, run `predicting/performance_evaluation/flexible_label/fault_tolerance/fault_tolerance.py` to evaluate the fault tolerance of flexible and specified labels.

    ```bash
    python  predicting/performance_evaluation/flexible_label/fault_tolerance/fault_tolerance.py
    ```

### Prediction of IVA HA RBD of hosts other than human and avian

1. Run `vBERT_and_GIVAL/GIVAL/IAV_predicting_for_test_final_other_host_predicting.py` for host adaptation prediction of IVA HA RBD of hosts except for human and avian.

    ```bash
    python  vBERT_and_GIVAL/GIVAL/IAV_predicting_for_test_final_other_host_predicting.py
    ```

2. Run `predicting/other_host_predicting/scatterpie_subtype_other_host.py` and `predicting/other_host_predicting/other_host_heatmap_human_ratio_with_weight_continent_year.py` to analyze the adaptation prediction results.

    ```bash
    python  predicting/other_host_predicting/scatterpie_subtype_other_host.py
    python  predicting/other_host_predicting/other_host_heatmap_human_ratio_with_weight_continent_year.py
    ```

3. Run `predicting/other_host_predicting/Bayes_analysis/logo_plot_by_aminos_H3N2.py` and `predicting/other_host_predicting/Bayes_analysis/logo_plot_by_aminos_H3N8.py` to analyze the important site of H3N2 and H3N8 HA.

    ```bash
    python  predicting/other_host_predicting/Bayes_analysis/logo_plot_by_aminos_H3N2.py
    python  predicting/other_host_predicting/Bayes_analysis/logo_plot_by_aminos_H3N8.py
    ```

### Structure prediction of H3N2 and H3N8 HA RBD

1. Run `predicting/structure_prediction/esmfold_H3N2.py`, `predicting/structure_prediction/esmfold_H3N2_human.py`, `predicting/structure_prediction/esmfold_H3N8.py` and `predicting/structure_prediction/esmfold_H3N2_H3N8_for_structure_alignment.py` for protein structure prediction of selected H3N2 and H3N8 HA RBD sequences based on ESMfold. Make sure you have copied the py files and all of the folders to the package of ESMfold to run the py files.

    ```bash
    python predicting/structure_prediction/esmfold_H3N2.py
    python predicting/structure_prediction/esmfold_H3N2_human.py
    python predicting/structure_prediction/esmfold_H3N8.py
    python predicting/structure_prediction/esmfold_H3N2_H3N8_for_structure_alignment.py
    ```

2. Copy commands of txt files in `predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/pymol_command_python_generated/` to pymol to calculate RMSD values.

3. Run `predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N2/extract_H3N2_with_canine.py`, `predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N2/extract_H3N2_with_human.py`, `predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N2/extract_H3N2_with_swine.py`, `predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N2/for_ridge_plot_H3N2_all_reshape.py` and `predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N2/KW_test.py`; `predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N8/extract_H3N8_with_canine.py`, `predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N8/extract_H3N8_with_equine.py`, `predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N8/extract_H3N8_with_human.py`, `predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N8/for_ridge_plot_H3N8_all_reshape.py`, `predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N8/KW_test.py` to further analyze the RMSD values of H3N2 and H3N8.

    ```bash
    python predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N2/extract_H3N2_with_canine.py
    python predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N2/extract_H3N2_with_human.py
    python predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N2/extract_H3N2_with_swine.py
    python predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N2/for_ridge_plot_H3N2_all_reshape.py
    predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N2/KW_test.py
    python predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N8/extract_H3N8_with_canine.py
    predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N8/extract_H3N8_with_equine.py
    predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N8/extract_H3N8_with_human.py
    python predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N8/for_ridge_plot_H3N8_all_reshape.py
    predicting/structure_prediction/H3N2_H3N8_pdb_after_pymol/H3N8/KW_test.py
    ```

### Prediction of 10 genes of monkeypox sequences

1. Run `predicting/monkeypox_predicting/tokenization/1tokenization_for_10gene_gisaid_trainset.py`, `predicting/monkeypox_predicting/tokenization/1tokenization_for_10gene_gisaid_validset.py`, `predicting/monkeypox_predicting/tokenization/1tokenization_for_10gene_ncbi_validset.py`, `predicting/monkeypox_predicting/tokenization/1tokenization_for_sampling_10gene_ncbi_valid_all1.py` and `predicting/monkeypox_predicting/tokenization/1tokenization_for_sampling_10gene_ncbi_valid_all2.py` for sequence tokenization of training, validation and predicing dataset.

    ```bash
    python predicting/monkeypox_predicting/tokenization/1tokenization_for_10gene_gisaid_trainset.py
    python predicting/monkeypox_predicting/tokenization/1tokenization_for_10gene_gisaid_validset.py
    python predicting/monkeypox_predicting/tokenization/1tokenization_for_10gene_ncbi_validset.py
    python predicting/monkeypox_predicting/tokenization/1tokenization_for_sampling_10gene_ncbi_valid_all1.py
    python predicting/monkeypox_predicting/tokenization/1tokenization_for_sampling_10gene_ncbi_valid_all2.py
    ```

2. Extract embeddings for the txt files generated with the above commands and rename the npy files of vBERT-optimized embeddings with the same name as the txt files. Move the npy files to `predicting/monkeypox_predicting/predicting/npy_file`.

3. Run `predicting/monkeypox_predicting/tokenization/2tokenization_pca_evo_shift_10gene.py` for sequence tokenization of unsupervised learning dataset.

    ```bash
    python predicting/monkeypox_predicting/tokenization/2tokenization_pca_evo_shift_10gene.py
    ```

4. Extract embeddings for the txt files generated with the above commands and rename the npy files of vBERT-optimized embeddings with the same name as the txt files. Move the npy files to `predicting/monkeypox_predicting/tokenization/npy_file`.

5. Run `predicting/monkeypox_predicting/tokenization/3pca_10gene_evo_shift.py` to analyze the adaptation shift of clade I and II.

    ```bash
    python predicting/monkeypox_predicting/tokenization/3pca_10gene_evo_shift.py
    ```
6. Run `predicting/monkeypox_predicting/predicting/1padding_train_set.py`, `predicting/monkeypox_predicting/predicting/2padding_valid_set.py`, `predicting/monkeypox_predicting/3_1padding_NCBI_all1.py` and `predicting/monkeypox_predicting/predicting/3_2padding_NCBI_all2.py` for padding of embedding and iformation extraction.

    ```bash
    python predicting/monkeypox_predicting/predicting/1padding_train_set.py
    python predicting/monkeypox_predicting/predicting/2padding_valid_set.py
    python predicting/monkeypox_predicting/predicting/3_1padding_NCBI_all1.py
    python predicting/monkeypox_predicting/predicting/3_2padding_NCBI_all2.py
    ```

7. Run `predicting/monkeypox_predicting/predicting/4Mpox_predict_model_establish_and_valid.py` and `predicting/monkeypox_predicting/predicting/5Mpox_predict_NCBI_all.py` for model establishment and predicting of input gene of monkeypox. The 10 genes are OPG002, OPG015, OPG019, OPG031, OPG034, OPG049, OPG100, OPG130, OPG170 and OPG172.

    ```bash
    python predicting/monkeypox_predicting/predicting/4Mpox_predict_model_establish_and_valid.py
    python predicting/monkeypox_predicting/predicting/5Mpox_predict_NCBI_all.py
    ```


