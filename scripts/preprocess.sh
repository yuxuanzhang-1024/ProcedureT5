SRC_TRAIN = '../dataset/Pistachio_example/transformer/src-train.txt'
TGT_TRAIN = '../dataset/Pistachio_example/transformer/tgt-train.txt'
SRC_VALID = '../dataset/Pistachio_example/transformer/src-valid.txt'
TGT_VALID = '../dataset/Pistachio_example/transformer/tgt-valid.txt'
SAVE_DATA = '../dataset/Pistachio_example/transformer/preprocessed'


/path/the/smiles2actions/enviroment/bin/onmt_preprocess \
    -train_src $SRC_TRAIN -train_tgt $TGT_TRAIN \
    -valid_src $SRC_VALID -valid_tgt $TGT_VALID \
    -save_data $SAVE_DATA -src_seq_length 300 -tgt_seq_length 300 \
    -src_vocab_size 2000 -tgt_vocab_size 2000 -shard_size 2000000