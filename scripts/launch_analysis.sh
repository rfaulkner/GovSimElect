#!/bin/bash

#SBATCH --account=aip-rgrosse
#SBATCH --job-name=gse_analysis
#SBATCH --output=slurm/output/%j_%x.out
#SBATCH --time=0-1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G

analysis_dir="/home/$USER/projects/aip-rgrosse/rfaulk/GovSimElect/llm_judge"
module load python/3.11.5
source .venv/bin/activate
cd $analysis_dir
python3 examples/leader_sentiment.py

