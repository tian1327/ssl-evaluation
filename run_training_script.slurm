#!/bin/bash

## ENVIRONMENT SETTINGS
#SBATCH --get-user-env=L     #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=semi_aves_supervised_in_imagenet_oracle_retrieved
#SBATCH --time=0-20:00          
#SBATCH --ntasks=12                
#SBATCH --ntasks-per-node=12 
#SBATCH --nodes=1      
#SBATCH --mem=60G  
#SBATCH --output=slurm_log/%x.%j                   
#SBATCH --gres=gpu:a100:1             # Request 1 GPUs per node
##SBATCH --gres=gpu:rtx:1 
##SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu 

##SBATCH --mail-type=ALL 
##SBATCH --mail-user=anwesha.basu@tamu.edu

module load Anaconda3
source /sw/eb/sw/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate swat

python -u run_train.py \
    --task semi_aves \
    --init imagenet \
    --alg supervised \
    --unlabel in \
    --num_iter 25000 \
    --lr 1e-3 \
    --wd 1e-4 \
    --exp_dir semi_aves_supervised_in_imagenet_oracle_retrieved \
    --MoCo false \
    --batch_size 32 \
    --retrieval_split T2T500+T2I0.25 \
    --trainval_un_in_oracle 
    

conda deactivate