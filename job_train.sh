#! /bin/bash
#SBATCH --job-name=chagas_pred
#SBATCH -D .
#SBATCH --partition=gpu
#SBATCH --account=unlimited
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH  --gpus=1
#SBATCH --output=batch_logs/%j.out
#SBATCH --error=batch_logs/%j.err

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

module load lang/Miniconda3/24.7.1-0
conda init
module load system/CUDA/12.9.1

# Initialize conda for bash shell
conda init bash
source ~/.bashrc  # This reloads the shell to apply conda settings

export TQDM_MININTERVAL=10
export PYTORCH_ALLOC_CONF=expandable_segments:True

source activate chagas_pred

python main.py --method lstm-attn

