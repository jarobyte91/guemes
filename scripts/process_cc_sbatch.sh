#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --account=rrg-emilios
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=25
#SBATCH --mem=50G
#SBATCH --mail-user=jarobyte91@gmail.com
#SBATCH --mail-type=ALL
srun python process_compute_canada.py
