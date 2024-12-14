#!/bin/bash
#SBATCH --job-name=train-chemical-extractor
#SBATCH -t 96:00:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p gpu3090                  # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH --gres=gpu:8                   # take 1 GPU, see https://www.hlrn.de/doc/display/PUB/GPU+Usage for more options
#SBATCH --mem-per-gpu=96G             # setting the right constraints for the splitted gpu partitions
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks-per-node=8           # number of tasks per node
#SBATCH --cpus-per-task=4            # number cores per task


conda info --envs
source activate test # Or whatever you called your environment.
conda info --env

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"
# google-t5/t5-base
# GT4SD/multitask-text-and-chemistry-t5-base-augm
# laituan245/molt5-base
srun python cli_trainer.py --model_name_or_path laituan245/molt5-base \
    --lr 3e-4 \
    --lr_decay 0.99 \
    --display_mode 'local' \
    --train_loss_log_interval 10 \
    --batch_size 8 \
    --accumulate_grad_batches 2 \
    --train_file ../dataset/OpenExp/T5/train.jsonl \
    --validation_file ../dataset/OpenExp/T5/valid.jsonl \
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
    --dirpath '/scratch/PI/hanyugao/yuxuan/results/molT5-base-openexp-ignore-pad' \
    --save_dir '/scratch/PI/hanyugao/yuxuan/results' \
    # --ckpt_path ../results/molt5-base-ignore-pad/epoch=18-step=40000.ckpt  \