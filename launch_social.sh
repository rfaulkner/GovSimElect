#!/bin/bash

#SBATCH --account=aip-rgrosse
#SBATCH --job-name=govsim-social
#SBATCH --output=slurm/output/%j_%x.out
#SBATCH --time=0-1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G

project_dir="/home/$USER/projects/aip-rgrosse/$USER/GovSimElect"

module load python/3.11.5 cuda/12.2 gcc arrow/21.0.0

cd $project_dir
source .venv/bin/activate
python3 -m simulation.analysis.social
