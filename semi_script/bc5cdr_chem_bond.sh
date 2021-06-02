#!/bin/bash

if [[ $# -ne 1 ]]; then
  GPUID=0
else
  GPUID=$1
fi

echo "Run on GPU $GPUID"

# data
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
DATA_ROOT=$PROJECT_ROOT/dataset/BC5CDR-chem/

# model
MODEL_TYPE=bert
MODEL_NAME=dmis-lab/biobert-base-cased-v1.1

# params
LR=1e-5
WEIGHT_DECAY=1e-4
EPOCH=20
SEED=0

ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98
WARMUP=200

TRAIN_BATCH=64
EVAL_BATCH=64

# self-training parameters
REINIT=0
BEGIN_STEP=2000
LABEL_MODE=soft
PERIOD=1000
# For HP_LABEL in [7,9], BOND will overwrite self-training labels by strong labels. (Only work for soft pseudo labels)
HP_LABEL=7.9

# output
OUTPUT=$PROJECT_ROOT/outputs/BC5CDR-chem/BOND/${MODEL_TYPE}_reinit${REINIT}_begin${BEGIN_STEP}_period${PERIOD}_${LABEL_MODE}_hp${HP_LABEL}_${EPOCH}_${LR}/

[ -e $OUTPUT/script  ] || mkdir -p $OUTPUT/script
cp -f $(readlink -f "$0") $OUTPUT/script
rsync -ruzC --exclude-from=$PROJECT_ROOT/.gitignore --exclude 'dataset' --exclude 'pretrained_model' --exclude 'outputs' $PROJECT_ROOT/ $OUTPUT/src

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 run_self_training_ner.py --data_dir $DATA_ROOT \
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
  --overwrite_output_dir \
  --self_training_reinit $REINIT --self_training_begin_step $BEGIN_STEP \
  --self_training_label_mode $LABEL_MODE --self_training_period $PERIOD \
  --self_training_hp_label $HP_LABEL \
  --load_weak 
