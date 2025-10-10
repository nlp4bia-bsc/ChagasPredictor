#!/bin/bash
#SBATCH --job-name=predict_train
#SBATCH -D .
#SBATCH -A bsc14
#SBATCH --qos=acc_debug
#SBATCH --output=batch_logs/%j.out
#SBATCH --error=batch_logs/%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:00

## --qos=acc_bscls
module load anaconda

# Initialize conda for bash shell
conda init bash
source ~/.bashrc  # This reloads the shell to apply conda settings

export TQDM_MININTERVAL=10

conda activate /gpfs/scratch/bsc14/.conda/envs/bertfine

/gpfs/scratch/bsc14/.conda/envs/bertfine/bin/python main.py --method lstm 

