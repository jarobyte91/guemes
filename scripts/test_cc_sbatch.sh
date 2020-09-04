#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --account=rrg-emilios
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=512M
#SBATCH --mail-user=jarobyte91@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
python compute_canada_test.py
