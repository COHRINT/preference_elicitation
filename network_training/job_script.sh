#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END
#SBATCH --mail-user=hura1154
#SBATCH --time=00:25:00
#SBATCH --partition=shas
#SBATCH --job-name=vae-trainer
#SBATCH --output=vae-trainer-%j.out

module purge
module load julia/1.6.0
module load cuda/10.1

echo
julia ./preference_elicitation/network_training/package_loader.jl
julia ./preference_elicitation/Interaction.jl
echo
