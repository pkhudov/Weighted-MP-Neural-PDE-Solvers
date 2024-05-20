#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# -N sigma-0.01-mp-pde-pavel
# Runtime limit of 36 hours:
#$ -l h_rt=36:00:00
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
# Request one GPU in the gpu queue:
#$ -q gpu
#$ -pe gpu-a100 1
#
# Request 32 GB system RAM
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -l h_vmem=32G

# Initialise the environment modules and load CUDA version 11.0.2
. /etc/profile.d/modules.sh
module load cuda/12.1

# Activate the conda environment
module load anaconda
source activate weighted-gnn-pavel

# Experiment E1: Burger's equation (0.5, 0, 0)
# 2096 trajectories
# Base Resoultion: 100
# Super Resolution: 200
# Temporal Resolution: 250
# Learning Rate: 1e-3
# Weight Decay: 1e-8
# Epochs: 20
# Batch Size: 16
# Hidden size: 164
# Maximum unrolling: 2
# Neighbours: 6

python experiments/train.py --device=cuda:0 --experiment=E1 --neighbors=6 --unrolling=2 --sigma=0.01 --lr=1e-3 --log=True
