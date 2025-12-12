#!/bin/bash

#SBATCH --account=aip-rgrosse
#SBATCH --job-name=govsim_elect
#SBATCH --output=slurm/output/%j_%x.out

#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G

# Gemma-3
# model_id="openrouter-google/gemma-3-27b-it"
# model_id="openrouter-google/gemma-3-12b-it:free"
# model_id="openrouter-google/gemini-2.5-flash"

# Llama-3
# model_id="openrouter-meta-llama/llama-3-8b-instruct"
model_id="openrouter-meta-llama/llama-3-70b-instruct"

# QWEN-2.5
# model_id="openrouter-qwen/qwen-2.5-72b-instruct:free"

# GPT
# model_id="gpt-4-turbo-2024-04-09"
# model_id="gpt-3.5-turbo-0125"
# model_id="gpt-4o-2024-05-13"

# Claude
# model_id="openrouter-anthropic/claude-3.5-sonnet"
# model_id="openrouter-anthropic/claude-3.5-haiku"

experiment="fish_baseline_concurrent_12_agents"

project_dir="/home/$USER/projects/aip-rgrosse/$USER/GovSimElect"

export WANDB_DISABLED=true # Optional, depending on if you want to use WandB
export HF_HOME="/scratch/$USER/hf_cache"

module load python/3.11.5 cuda/12.2 gcc arrow/21.0.0

cd $project_dir
source .venv/bin/activate

python3 -m simulation.main experiment=$experiment llm.path=$model_id llm.is_api=true
