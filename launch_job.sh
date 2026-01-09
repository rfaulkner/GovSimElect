#!/bin/bash

#SBATCH --account=aip-rgrosse
#SBATCH --output=slurm/output/%j_%x.out

#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G

# model_id="mistralai/Mistral-7B-Instruct-v0.2"
# model_id="meta-llama/Meta-Llama-3-70B-Instruct"
# model_id="meta-llama/Meta-Llama-3-8B-Instruct"
# model_id="Qwen/Qwen1.5-72B-Chat-GPTQ-Int4"
model_id="Qwen/Qwen1.5-110B-Chat-GPTQ-Int4"

experiment="fish_baseline_concurrent_leaders"

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

python3 -m simulation.main_elect experiment=$experiment llm.path=$model_id experiment.seed=$ARG1 experiment.agent.leader_population=$ARG2 experiment.env.disinformation=$ARG3
