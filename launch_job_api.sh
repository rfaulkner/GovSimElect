#!/bin/bash

#SBATCH --account=aip-rgrosse
#SBATCH --job-name=govsim_elect
#SBATCH --output=slurm/output/%j_%x.out

#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G

# Gemma-3 / Gemini
# model_id="openrouter-google/gemma-3-27b-it"
# model_id="openrouter-google/gemma-3-12b-it:free"
model_id="openrouter-google/gemini-2.5-flash"

# Llama-3
# model_id="openrouter-meta-llama/llama-3-8b-instruct"
# model_id="openrouter-meta-llama/llama-3-70b-instruct"

# QWEN-2.5
# model_id="openrouter-qwen/qwen-2.5-72b-instruct:free"

# GPT
# model_id="gpt-4-turbo-2024-04-09"
# model_id="gpt-3.5-turbo-0125"
# model_id="gpt-4o-2024-05-13"

# Claude
# model_id="openrouter-anthropic/claude-3.5-sonnet"
# model_id="openrouter-anthropic/claude-3.5-haiku"

experiment="fish_baseline_concurrent_leaders"
# experiment="fish_baseline_concurrent_leaders_51"

project_dir="/home/$USER/projects/aip-rgrosse/$USER/GovSimElect"

export WANDB_DISABLED=true # Optional, depending on if you want to use WandB
export HF_HOME="/scratch/$USER/hf_cache"

module load python/3.11.5 cuda/12.2 gcc arrow/21.0.0

cd $project_dir
source .venv/bin/activate

# Check if the first argument ($1) is empty
if [ -z "$1" ]; then
  # If empty, assign a default value to a new variable
  ARG1=1
else
  # Otherwise, use the provided argument
  ARG1="$1"
fi

# Check if the first argument ($2) is empty
if [ -z "$2" ]; then
  # If empty, assign a default value to a new variable
  ARG2=balanced
else
  # Otherwise, use the provided argument
  ARG2="$2"
fi

# Check if the first argument ($3) is empty
if [ -z "$1" ]; then
  # If empty, assign a default value to a new variable
  ARG3=true
else
  # Otherwise, use the provided argument
  ARG3="$3"
fi

python3 -m simulation.main_elect experiment=$experiment llm.path=$model_id llm.is_api=true experiment.seed=$ARG1 experiment.agent.leader_population=$ARG2 experiment.env.disinformation=$ARG3
