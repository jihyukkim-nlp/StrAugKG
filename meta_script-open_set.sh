#!/bin/bash
# We used preprocessed data from https://github.com/Chen-Wang-CUHK/KG-KE-KR-M
wget https://www.dropbox.com/s/lgeza7owhn9dwtu/Processed_data_for_onmt.zip?dl=1
unzip Processed_data_for_onmt.zip\?dl\=1 
rm Processed_data_for_onmt.zip\?dl\=1 

mkdir -p data/kp20k_filtered
# train data
cp data/Processed_data_for_onmt/Training/word_kp20k_training_context_filtered.txt data/kp20k_filtered/train_src.txt
cp data/Processed_data_for_onmt/Training/word_kp20k_training_context_nstpws_sims_retrieved_keyphrases_filtered.txt data/kp20k_filtered/train_ret.txt
cp data/Processed_data_for_onmt/Training/word_kp20k_training_keyword_filtered.txt data/kp20k_filtered/train_trg.txt
# validation data
cp data/Processed_data_for_onmt/Validation/word_kp20k_validation_context_filtered.txt data/kp20k_filtered/valid_src.txt
cp data/Processed_data_for_onmt/Validation/word_kp20k_validation_context_nstpws_sims_retrieved_keyphrases_filtered.txt data/kp20k_filtered/valid_ret.txt
cp data/Processed_data_for_onmt/Validation/word_kp20k_validation_keyword_filtered.txt data/kp20k_filtered/valid_trg.txt
# test data
cp data/Processed_data_for_onmt/Testing/word_kp20k_testing_context.txt data/kp20k_filtered/test_src.txt
cp data/Processed_data_for_onmt/Testing/word_kp20k_testing_context_nstpws_sims_retrieved_keyphrases_filtered.txt data/kp20k_filtered/test_ret.txt
cp data/Processed_data_for_onmt/Testing/word_kp20k_testing_keyword.txt data/kp20k_filtered/test_trg.txt

rm -r data/Processed_data_for_onmt*

# Preprocessing
# w/o title
cd open_set
python preprocess.py -data_dir ../data/kp20k_filtered -vocab_size 50000
# w/ title
mkdir -p ../data/kp20k_filtered_title
cp ../data/kp20k_filtered/*.txt ../data/kp20k_filtered_title/
python preprocess.py -data_dir ../data/kp20k_filtered_title -vocab_size 50000 -use_title
cd ../

# run experiment
cd open_set
# w/o title
source run.sh 0 3 
# w/ title
source run.sh 0 3 title
cd ../
