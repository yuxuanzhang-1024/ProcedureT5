#!/bin/bash
#SBATCH --job-name=ProcedureT5-Train
#SBATCH -t 96:00:00                 
#SBATCH -p <YOUR_PARTITION>               
#SBATCH --gres=gpu:8   
#SBATCH --mem-per-gpu=96G        
#SBATCH --nodes=1                
#SBATCH --ntasks-per-node=8        
#SBATCH --cpus-per-task=4            


conda info --envs
source activate ProcedureT5
conda info --env

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

srun python ../ProcedureT5/train/cli_trainer.py --model_name_or_path laituan245/molt5-base \
    --lr 3e-4 \
    --lr_decay 0.99 \
    --display_mode 'local' \
    --train_loss_log_interval 10 \
    --batch_size 8 \
    --accumulate_grad_batches 2 \
    --train_file ../dataset/YOUR/PATH/TO/DATASET/train.jsonl \
    --validation_file ../dataset/YOUR/PATH/TO/DATASET/valid.jsonl \
    --type cgm \
    --val_check_interval 2000  \
    --max_epochs 50 \
    --limit_val_batches 1000 \
    --log_every_n_steps 200 \
    --devices 8 \
    --monitor val_acc_100 \
    --save_top_k 3 \
    --mode max \
    --every_n_train_steps 2000 \
    --accelerator 'auto' \
    --dirpath '/DIRPATH/TO/SAVE/CHECKPOINTS' \