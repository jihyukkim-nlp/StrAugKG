#!/bin/bash
wget https://www.dropbox.com/s/lgeza7owhn9dwtu/Processed_data_for_onmt.zip?dl=1
unzip Processed_data_for_onmt.zip\?dl\=1 
rm Processed_data_for_onmt.zip\?dl\=1 
# data/Processed_data_for_onmt/Training/word_kp20k_training_context_filtered.txt
# data/Processed_data_for_onmt/Training/word_kp20k_training_context_nstpws_sims_retrieved_keyphrases_filtered.txt
# data/Processed_data_for_onmt/Training/word_kp20k_training_keyword_filtered.txt

mkdir -p data/kp20k_filtered
# train
cp data/Processed_data_for_onmt/Training/word_kp20k_training_context_filtered.txt data/kp20k_filtered/train_src.txt
cp data/Processed_data_for_onmt/Training/word_kp20k_training_context_nstpws_sims_retrieved_keyphrases_filtered.txt data/kp20k_filtered/train_ret.txt
cp data/Processed_data_for_onmt/Training/word_kp20k_training_keyword_filtered.txt data/kp20k_filtered/train_trg.txt
# valid
cp data/Processed_data_for_onmt/Validation/word_kp20k_validation_context_filtered.txt data/kp20k_filtered/valid_src.txt
cp data/Processed_data_for_onmt/Validation/word_kp20k_validation_context_nstpws_sims_retrieved_keyphrases_filtered.txt data/kp20k_filtered/valid_ret.txt
cp data/Processed_data_for_onmt/Validation/word_kp20k_validation_keyword_filtered.txt data/kp20k_filtered/valid_trg.txt
# test
cp data/Processed_data_for_onmt/Testing/word_kp20k_testing_context.txt data/kp20k_filtered/test_src.txt
cp data/Processed_data_for_onmt/Testing/word_kp20k_testing_context_nstpws_sims_retrieved_keyphrases_filtered.txt data/kp20k_filtered/test_ret.txt
cp data/Processed_data_for_onmt/Testing/word_kp20k_testing_keyword.txt data/kp20k_filtered/test_trg.txt

# Preprocessing
# w/o title
cd scientific_publications
python preprocess.py -data_dir ../data/kp20k_filtered -vocab_size 50000
# w/ title
mkdir -p ../data/kp20k_filtered_title
cp ../data/kp20k_filtered/*.txt ../data/kp20k_filtered_title/
python preprocess.py -data_dir ../data/kp20k_filtered_title -vocab_size 50000 -use_title
cd ../

# run experiment
# w/o title
source run.sh 0 3 
# w/ title
source run.sh 0 3 title
