SRC_TRAIN = '' # Path to the training source data
TGT_TRAIN = '' # Path to the training target data
SRC_VALID = '' # Path to the validation source data
TGT_VALID = '' # Path to the validation target data
SAVE_DATA = '' # Path to save the preprocessed data


onmt_preprocess \
    -train_src $SRC_TRAIN -train_tgt $TGT_TRAIN \
    -valid_src $SRC_VALID -valid_tgt $TGT_VALID \
    -save_data $SAVE_DATA -src_seq_length 300 -tgt_seq_length 300 \
    -src_vocab_size 2000 -tgt_vocab_size 2000 -shard_size 2000000