#!/bin/bash

#SBATCH --nodes=1
#SBATCH --exclude=vine3
##SBATCH --ntasks=1
#SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=0:15:00
#SBATCH --array=0
#SBATCH --job-name=evaluate
#SBATCH --output=evaluate_%A_%a.out

module purge
module load cuda-10.0
source /home/eo41/venv/bin/activate

python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_validation.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_l2_ns' --im-size 800
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_validation.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'resnext101_32x48d_wsl'

echo "Done"
