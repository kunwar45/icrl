#!/bin/bash
#SBATCH --job-name=icrl-finetune
#SBATCH --account=def-srirams
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

module load python/3.11 cuda/12.1 cudnn/8.9

source ~/envs/icrl/bin/activate

cd $SLURM_SUBMIT_DIR

python scripts/run_finetune.py "$@"
