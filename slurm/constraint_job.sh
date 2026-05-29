#!/bin/bash
#SBATCH --job-name=icrl-constraint
#SBATCH --account=def-srirams
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

module load python/3.11 cuda/12.1 cudnn/8.9

source ~/envs/icrl/bin/activate

cd $SLURM_SUBMIT_DIR

python scripts/train_constraint.py "$@"
