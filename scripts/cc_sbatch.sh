#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --account=rrg-emilios
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem=10G
#SBATCH --mail-user=jarobyte91@gmail.com
#SBATCH --mail-type=ALL

srun python test_cc_launch_experiments.py
