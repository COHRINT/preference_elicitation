#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=aa100-ucb
#SBATCH --mail-type=END
#SBATCH --mail-user=hura1154
#SBATCH --time=5:00:00
#SBATCH --job-name=py-vae-trainer
#SBATCH --output=py-vae-trainer-1-%j.out


module load anaconda
module load cuda/11.2
conda activate vae_alpine
python network_training/atlas_vae.py --lr 1e-4 --beta 1
