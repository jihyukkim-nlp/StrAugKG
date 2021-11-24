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

CUDA_VISIBLE_DEVICES=$device python train.py -data ../data/$dataset/ \
$train_opt $model_opt \
-exp_path exp/ -exp $exp 