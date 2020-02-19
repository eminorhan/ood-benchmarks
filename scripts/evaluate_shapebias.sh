#!/bin/bash

#SBATCH --nodes=1
#SBATCH --exclude=vine3,vine8,vine11,vine12,vine13
##SBATCH --ntasks=1
#SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=6:00:00
#SBATCH --array=0
#SBATCH --job-name=shapebias
#SBATCH --output=shapebias_%A_%a.out

index=$SLURM_ARRAY_TASK_ID
job=$SLURM_JOBID

module purge

module load cuda-10.0
source /home/eo41/venv/bin/activate

#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_shapebias.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/shape_bias/' --model-name 'tf_efficientnet_l2_ns' --im-size 800
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_shapebias.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/shape_bias/' --model-name 'tf_efficientnet_l2_ns_475' --im-size 475
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_shapebias.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/shape_bias/' --model-name 'tf_efficientnet_b7_ns' --im-size 600
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_shapebias.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/shape_bias/' --model-name 'tf_efficientnet_b6_ns' --im-size 528
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_shapebias.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/shape_bias/' --model-name 'tf_efficientnet_b5_ns' --im-size 456
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_shapebias.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/shape_bias/' --model-name 'tf_efficientnet_b4_ns' --im-size 380
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_shapebias.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/shape_bias/' --model-name 'tf_efficientnet_b3_ns' --im-size 300
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_shapebias.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/shape_bias/' --model-name 'tf_efficientnet_b2_ns' --im-size 260
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_shapebias.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/shape_bias/' --model-name 'tf_efficientnet_b1_ns' --im-size 240
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_shapebias.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/shape_bias/' --model-name 'tf_efficientnet_b0_ns' --im-size 224
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_shapebias.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/shape_bias/' --model-name 'resnext101_32x48d_wsl'
python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_shapebias.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/shape_bias/' --model-name 'resnext101_32x32d_wsl'
python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_shapebias.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/shape_bias/' --model-name 'resnext101_32x16d_wsl'
python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_shapebias.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/shape_bias/' --model-name 'resnext101_32x8d_wsl'
python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_shapebias.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/shape_bias/' --model-name 'resnext101_32x8d'

echo "Done"
