#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END
#SBATCH --mail-user=hura1154
#SBATCH --time=20:00:00
#SBATCH --partition=sgpu
#SBATCH --job-name=vae-trainer
#SBATCH --output=vae-trainer-%j.out

module purge
module load julia/1.6.6
module load cuda/11.2

echo
julia ./network_training/vae_train.jl
echo
