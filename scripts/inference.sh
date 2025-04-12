#!/bin/bash

# Set paths for data
SRC_PATH='' # Path to the source test data
TGT_PATH='' # Path to the target test data
PRED_PATH='' # Path to save the predictions

# Set the GPU device to use
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH='' # Path to the trained model

onmt_translate \
    -model $MODEL_PATH \
    -src $SRC_PATH \
    -tgt $TGT_PATH \
    -output $PRED_PATH \
    -verbose \
    -max_length 400 \
    -batch_size 16 \
    -gpu 0

