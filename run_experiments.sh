#!/bin/sh
readonly MAX_EPOCH=250
readonly BATCH_SIZE=512
readonly NUM_MC=150
readonly MAX_RUL=125
readonly MAX_RUN_NUMBER=9

# FD001 *******************************************************************************************************************************************
DATASET="CMAPSS/FD001"
LOG_PATH="log/$DATASET/min-max"
DUMP_PATH="dump/$DATASET/min-max"

MODEL="bayesian_conv5_dense1"
QUANTITY=1.00
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --quantity $QUANTITY

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_conv5_dense1"
QUANTITY=0.50
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --quantity $QUANTITY

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_conv5_dense1"
QUANTITY=0.25
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --quantity $QUANTITY

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_conv5_dense1"
QUANTITY=0.125
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --quantity $QUANTITY

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_conv2_pool2"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_dense3"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="frequentist_conv5_dense1"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE
    done

MODEL="frequentist_conv2_pool2"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE
    done

MODEL="frequentist_dense3"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE
    done

# FD002 *******************************************************************************************************************************************
DATASET="CMAPSS/FD002"
LOG_PATH="log/$DATASET/min-max"
DUMP_PATH="dump/$DATASET/min-max"

MODEL="bayesian_conv5_dense1"
QUANTITY=1.00
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --quantity $QUANTITY

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_conv5_dense1"
QUANTITY=0.50
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --quantity $QUANTITY

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_conv5_dense1"
QUANTITY=0.25
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --quantity $QUANTITY

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_conv5_dense1"
QUANTITY=0.125
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --quantity $QUANTITY

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_conv2_pool2"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_dense3"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="frequentist_conv5_dense1"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE
    done

MODEL="frequentist_conv2_pool2"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE
    done

MODEL="frequentist_dense3"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE
    done

# FD003 *******************************************************************************************************************************************
DATASET="CMAPSS/FD003"
LOG_PATH="log/$DATASET/min-max"
DUMP_PATH="dump/$DATASET/min-max"

MODEL="bayesian_conv5_dense1"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_conv2_pool2"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_dense3"
QUANTITY=1.00
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --quantity $QUANTITY

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_dense3"
QUANTITY=0.50
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --quantity $QUANTITY

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_dense3"
QUANTITY=0.25
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --quantity $QUANTITY

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_dense3"
QUANTITY=0.125
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --quantity $QUANTITY

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="frequentist_conv5_dense1"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE
    done

MODEL="frequentist_conv2_pool2"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE
    done

MODEL="frequentist_dense3"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE
    done

# FD004 *******************************************************************************************************************************************
DATASET="CMAPSS/FD004"
LOG_PATH="log/$DATASET/min-max"
DUMP_PATH="dump/$DATASET/min-max"

MODEL="bayesian_conv5_dense1"
QUANTITY=1.00
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --quantity $QUANTITY

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_conv5_dense1"
QUANTITY=0.50
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --quantity $QUANTITY

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_conv5_dense1"
QUANTITY=0.25
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --quantity $QUANTITY

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_conv5_dense1"
QUANTITY=0.125
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL --quantity $QUANTITY

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$QUANTITY/${MODEL}_${QUANTITY}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_conv2_pool2"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="bayesian_dense3"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE --num_mc $NUM_MC
    done

MODEL="frequentist_conv5_dense1"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE
    done

MODEL="frequentist_conv2_pool2"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE
    done

MODEL="frequentist_dense3"
for i in $(seq 0 $MAX_RUN_NUMBER)
    do
        python3 train.py --dataset $DATASET --model $MODEL --log_dir $LOG_PATH/$MODEL/${MODEL}_$i --max_epoch $MAX_EPOCH --batch_size $BATCH_SIZE --max_rul $MAX_RUL

        python3 evaluate.py --dataset $DATASET --model $MODEL --model_path $LOG_PATH/$MODEL/${MODEL}_$i/checkpoint.pth.tar --dump_dir $DUMP_PATH/$MODEL/${MODEL}_$i --batch_size $BATCH_SIZE
    done
