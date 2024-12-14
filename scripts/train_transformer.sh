DATA_PATH='../dataset/Pistachio_example/transformer/preprocessed'
MODEL_SAVE_PATH='../dataset/Pistachio_example/transformer/models/'
mkdir -p $MODEL_SAVE_PATH
LOG_PATH='../dataset/Pistachio_example/transformer/train_log.txt'

/home/yuxuan/anaconda3/envs/new/bin/onmt_train \
  -data $DATA_PATH  -save_model  $MODEL_SAVE_PATH  \
  -seed 42 -save_checkpoint_steps 10000 -keep_checkpoint 100 \
  -train_steps 1000000 -param_init 0  -param_init_glorot \
  -max_generator_batches 32 -batch_size 16 \
  -normalization tokens -max_grad_norm 0  -accum_count 4 \
  -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam \
  -warmup_steps 8000  -learning_rate 2 -label_smoothing 0.0 \
  -report_every 1000  -valid_batch_size 8 -layers 4 -rnn_size 256 \
  -word_vec_size 256 -encoder_type transformer -decoder_type transformer \
  -dropout 0.1 -position_encoding -valid_steps 10000 \
  -global_attention general -global_attention_function softmax \
  -self_attn_type scaled-dot -heads 8 -transformer_ff 2048 -gpu_ranks 0 1 -world_size 2\
  -log_file $LOG_PATH 