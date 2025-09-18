#!/bin/bash

#SBATCH --account=aip-rgrosse
#SBATCH --job-name=govsim_elect
#SBATCH --output=slurm/output/%j_%x.out

#SBATCH --time=0-1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G

# model_id="openrouter-google/gemma-3-27b-it:free"
model_id="openrouter-google/gemma-3-12b-it:free"

experiment="fish_baseline_concurrent_leaders"

project_dir="/home/$USER/projects/aip-rgrosse/$USER/GovSimElect"

export WANDB_DISABLED=true # Optional, depending on if you want to use WandB
export HF_HOME="/scratch/$USER/hf_cache"

module load python/3.11.5 cuda/12.2 gcc arrow/21.0.0

cd $project_dir
source .venv/bin/activate

python3 -m simulation.main_elect experiment=$experiment llm.path=$model_id llm.is_api=true
