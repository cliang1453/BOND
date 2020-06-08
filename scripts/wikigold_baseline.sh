#!/bin/bash

if [[ $# -ne 1 ]]; then
  GPUID=0
else
  GPUID=$1
fi

echo "Run on GPU $GPUID"

# data
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
DATA_ROOT=$PROJECT_ROOT/dataset/wikigold_distant/

# model
MODEL_TYPE=roberta
MODEL_NAME=roberta-base

# params
LR=1e-5
WEIGHT_DECAY=1e-4
EPOCH=20
SEED=0

ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98
WARMUP=70

TRAIN_BATCH=16
EVAL_BATCH=32

# output
OUTPUT=$PROJECT_ROOT/outputs/wikigold/baseline/${MODEL_TYPE}_${EPOCH}_${LR}/

[ -e $OUTPUT/script  ] || mkdir -p $OUTPUT/script
cp -f $(readlink -f "$0") $OUTPUT/script
rsync -ruzC --exclude-from=$PROJECT_ROOT/.gitignore --exclude 'dataset' --exclude 'pretrained_model' --exclude 'outputs' $PROJECT_ROOT/ $OUTPUT/src

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 run_ner.py --data_dir $DATA_ROOT \
  --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --adam_epsilon $ADAM_EPS \
  --adam_beta1 $ADAM_BETA1 \
  --adam_beta2 $ADAM_BETA2 \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --logging_steps 100 \
  --save_steps 100000 \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluate_during_training \
  --output_dir $OUTPUT \
  --cache_dir $PROJECT_ROOT/pretrained_model \
  --seed $SEED \
  --max_seq_length 128 \
  --overwrite_output_dir