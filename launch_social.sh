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

# Check if the first argument ($2) is empty
if [ -z "$1" ]; then
  # If empty, assign a default value to a new variable
  ARG1=balanced
else
  # Otherwise, use the provided argument
  ARG1="$1"
fi

# Check if the first argument ($3) is empty
if [ -z "$2" ]; then
  # If empty, assign a default value to a new variable
  ARG2=true
else
  # Otherwise, use the provided argument
  ARG2="$2"
fi

# Check if the first argument ($3) is empty
if [ -z "$3" ]; then
  # If empty, assign a default value to a new variable
  ARG3=Qwen/Qwen1.5-110B-Chat-GPTQ-Int4
else
  # Otherwise, use the provided argument
  ARG3="$3"
fi

python3 -m simulation.analysis.social $ARG1 $ARG2 $ARG3
