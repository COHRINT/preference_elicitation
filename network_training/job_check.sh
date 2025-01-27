#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END
#SBATCH --mail-user=hura1154
#SBATCH --time=20:00:00
#SBATCH --partition=shas
#SBATCH --job-name=job_test
#SBATCH --output=vae-trainer-%j.out

module purge
module load julia/1.6.0
module load cuda/11.2

echo
julia ./network_training/package_loader.jl
echo
