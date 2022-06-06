#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=hura1154
#SBATCH --time=00:10:00
#SBATCH --partition=shas
#SBATCH --job-name=vae-trainer
#SBATCH --output=vae-trainer-%j.out

module purge
module load julia/1.6.0
module load cuda/11.2

echo
julia
using Pkg
Pkg.add(["ImageIO","CategoricalArrays","CUDA","Flux","BSON","DrWatson","Images","Logging","Parameters","ProgressMeter","TensorBoardLogger","Random","MosaicViews","BasicPOMCP","POMDPs","ParticleFilters","Distributions","LinearAlgebra","Distances","Statistics","POMDPPolicies"])
exit()
pwd
julia ./projects/hura1154/preference_elicitation/Interaction.jl
echo
