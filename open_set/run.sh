#!/bin/bash
# ./train.sh [device] [enc_layers] [title]

device=$1
enc_layers=$2

if [[ $3 == "title" ]];then
    exp='o2o-title-gcn'$enc_layers'-nk'
    dataset=kp20k_filtered_title
    model_opt="-vocab ../data/"$dataset"/ -vocab_size 50000 -use_title"
else
    exp='o2o-gcn'$enc_layers'-nk'
    dataset=kp20k_filtered
    model_opt="-vocab ../data/"$dataset"/ -vocab_size 50000"
fi

model_opt=$model_opt" -word_vec_size 300 -enc_layers "$enc_layers
train_opt=" -epochs 20 -copy_attention -train_ml -batch_size 64 -seed 9527 -start_checkpoint_at 1"


# Train
CUDA_VISIBLE_DEVICES=$device python train.py -data ../data/$dataset/ \
$train_opt $model_opt \
-exp_path exp/ -exp $exp 


# # Prediction
# CUDA_VISIBLE_DEVICES=$device python interactive_predict.py \
# $model_opt \
# -vocab ../data/$dataset/ \
# -max_length 6 -remove_title_eos -n_best 30 -max_eos_per_output_seq 1 -beam_size 200 -batch_size 3 -replace_unk -copy_attention \
# -pred_path pred/ -exp $exp \
# \
# -src_file ../data/$dataset/test_src.txt \
# -model model/$exp/best.model


# # Evaluation
# pred_exp=predict.$exp
# python evaluate_prediction.py \
# -pred_file_path pred/$pred_exp/predictions.txt \
# -src_file_path ../data/$dataset/test_src.txt \
# -trg_file_path ../data/$dataset/test_trg.txt \
# -export_filtered_pred \
# -disable_extra_one_word_filter \
# -invalidate_unk \
# -all_ks 5 10 -present_ks 5 10 -absent_ks 5 10
# cat exp/$pred_exp/results_log_5_10_5_10_5_10.txt
