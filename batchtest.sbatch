#!/bin/bash
#SBATCH -JStableDiffusionBatch
#SBATCH -N1 --gres=gpu:A100:1 --ntasks-per-node=4
#SBATCH --mem-per-gpu=12G
#SBATCH -t10:00:00
#SBATCH -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nduggal3@gatech.edu

cd $HOME/scratch/controlnet-augmentation
source ~/.bashrc

conda activate ldm
srun makeimages.py
