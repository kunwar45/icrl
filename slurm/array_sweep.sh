#!/bin/bash
#SBATCH --job-name=icrl-sweep
#SBATCH --account=def-srirams
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

module load python/3.11 cuda/12.1 cudnn/8.9
source ~/envs/icrl/bin/activate
cd $SLURM_SUBMIT_DIR

EPSILONS=(0.05 0.1 0.2)
SEEDS=(42 123 456)

IDX=$SLURM_ARRAY_TASK_ID
EPS=${EPSILONS[$((IDX / 3))]}
SEED=${SEEDS[$((IDX % 3))]}
RUN_NAME="finetune_eps${EPS}_seed${SEED}"

python scripts/run_finetune.py \
    +compute=carleton \
    run_name=$RUN_NAME \
    finetune.constraint.epsilon=$EPS \
    seed=$SEED
