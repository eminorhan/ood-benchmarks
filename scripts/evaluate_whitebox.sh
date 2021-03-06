#!/bin/bash

#SBATCH --nodes=1
#SBATCH --exclude=hpc1,hpc2,hpc3,hpc4,hpc5,hpc6,hpc7,hpc8,hpc9,vine3,vine4,vine6,vine11,vine12,lion17,rose7,rose8,rose9
##SBATCH --ntasks=1
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=whi
#SBATCH --output=whi_%A_%a.out

module purge
module load cuda-10.0
source /home/eo41/venv/bin/activate

#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_l2_ns_475' --im-size 224
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b7_ns' --im-size 600
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b6_ns' --im-size 528
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b5_ns' --im-size 456
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b4_ns' --im-size 380
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b3_ns' --im-size 300
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b2_ns' --im-size 260
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b1_ns' --im-size 240
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b0_ns' --im-size 224

#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b8' --im-size 672
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b7' --im-size 600
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b6' --im-size 528
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b5' --im-size 456
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b4' --im-size 380
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b3' --im-size 300
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b2' --im-size 260
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b1' --im-size 240
#python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'tf_efficientnet_b0' --im-size 224

python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'moco_v2' 
python -u /misc/vlgscratch4/LakeGroup/emin/oos_benchmarks/evaluate_whitebox.py '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/' --model-name 'resnet50'

echo "Done"
