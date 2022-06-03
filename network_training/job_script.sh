#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=START
#SBATCH --mail-type=END
#SBATCH --mail-user=hura1154
#SBATCH --time=00:05:00
#SBATCH --partition=shas-testing
#SBATCH --job-name=vae-trainer
#SBATCH --output=vae-trainer-%j.out

module purge
module load julia/1.6.0
module load cuda/11.2

./projects/hura1154/preference_elicitation/network_training/vae_train.jl
