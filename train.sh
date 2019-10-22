#!/bin/sh
readonly DATASET="CMAPSS/FD001"
readonly MODEL="frequentist_dense3"
readonly LOG_PATH="log/$DATASET/min-max/$MODEL/${MODEL}_test"
readonly MAX_EPOCH=250
readonly BATCH_SIZE=512
readonly MAX_RUL=125
readonly NUM_MC=1

python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --num_mc $NUM_MC
