#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --mail-type=END
#SBATCH --mail-user=hura1154
#SBATCH --time=8:00:00
#SBATCH --job-name=py-vae-trainer
#SBATCH --output=py-vae-trainer-%j.out


module load anaconda
module load cuda/10.2
conda activate vae_python
python network_training/vanila_vae.py --lr 1e-4 --beta 1

