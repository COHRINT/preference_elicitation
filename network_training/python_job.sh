#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=sgpu
#SBATCH --mail-type=END
#SBATCH --mail-user=hura1154
#SBATCH --time=6:00:00
#SBATCH --job-name=py-vae-trainer
#SBATCH --output=py-vae-trainer-6-%j.out


module load anaconda
module load cuda/10.2
conda activate vae_python
python network_training/vanila_vae.py --nerve 32 --lvs 150 --beta 10 

