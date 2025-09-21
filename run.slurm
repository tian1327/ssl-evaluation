#!/bin/bash

## ENVIRONMENT SETTINGS
#SBATCH --get-user-env=L          # Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=aves_supervised_dinov2
##SBATCH --account=132713661936
#SBATCH --time=24:00:00
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --output=slurm_log/%x.%j
#SBATCH --gres=gpu:a100:1             # Request 1 GPUs per node
##SBATCH --gres=gpu:rtx:1 
##SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu          # Submit job to the gpu queue

##SBATCH --mail-type=ALL 
##SBATCH --mail-user=tian.liu@tamu.edu

##----------- execute python script
#python train.py --dataset semi_aves --num-labeled 5959 --arch resnet50 --init inat --batch-size 32 --mu 5 --lambda-u 1.0 --lr 0.001 --wdecay 1e-4 --threshold 0.8 --total-steps 50000 --expand-labels --seed 1 --out results/fixmatch --no-progress
#python train.py --dataset semi_aves --num-labeled 5959 --arch resnet50 --init imagenet --batch-size 32 --mu 5 --lambda-u 1.0 --lr 0.001 --wdecay 1e-4 --threshold 0.8 --total-steps 50000 --expand-labels --seed 1 --out results/fixmatch --no-progress


# use dinov2 backbone
python run_train.py --task semi_aves --model dinov2_vitb14_reg_lc --init scratch --alg supervised --unlabel in --num_iter 20000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in_dinov2 --MoCo false --trainval

# fixmatch
#python train.py --dataset semi_aves --num-labeled 5959 --arch dinov2_vitb14_reg_lc --init scratch --batch-size 32 --mu 1 --lambda-u 0.0 --lr 0.001 --wdecay 1e-4  --threshold 0.8 --total-steps 15000 --expand-labels --seed 1 --out results/aves_fixmatch_dinov2_vitb14_reg_lc_bs32_mu1_lmd0.0_thd0.8_lr1e-3_wd1e-4_iter15000

##----------- Job Environment variables
echo "Job ID: $SLURM_JOBID"
echo "Job submission directory: $SLURM_SUBMIT_DIR"
echo "Temp directory: $TMPDIR"
echo "Scratch directory: $SCRATCH"