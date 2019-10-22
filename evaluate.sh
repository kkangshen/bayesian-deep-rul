#!/bin/sh
readonly DATASET="CMAPSS/FD001"
readonly MODEL="frequentist_dense3"
readonly LOG_PATH="log/$DATASET/min-max/$MODEL/${MODEL}_test"
readonly DUMP_PATH="dump/$DATASET/min-max$MODEL/${MODEL}_test"
readonly BATCH_SIZE=512
readonly MAX_RUL=125
readonly NUM_MC=1

python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/checkpoint.pth.tar --dump_dir $DUMP_PATH --batch_size $BATCH_SIZE --num_mc $NUM_MC
