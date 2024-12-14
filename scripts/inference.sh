#!/bin/bash

# Set paths for data
SRC_PATH='../dataset/Pistachio_example/transformer/src-test.txt'
TGT_PATH='../dataset/Pistachio_example/transformer/tgt-test.txt'
PRED_PATH='../dataset/Pistachio_example/transformer/pred.txt'

# Set the GPU device to use
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH='../dataset/Pistachio_example/transformer/models/your_model_name'

/path/the/smiles2actions/enviroment/bin/onmt_translate \
    -model $MODEL_PATH \
    -src $SRC_PATH \
    -tgt $TGT_PATH \
    -output $PRED_PATH \
    -verbose \
    -max_length 400 \
    -batch_size 16 \
    -gpu 0

