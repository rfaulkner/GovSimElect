#!/bin/bash

#SBATCH --account=aip-rgrosse
#SBATCH --job-name=llm_judge
#SBATCH --output=slurm/output/%j_%x.out
#SBATCH --time=0-1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G

analysis_dir="/home/$USER/projects/aip-rgrosse/rfaulk/GovSimElect/llm_judge"
module load python/3.11.5
source .venv/bin/activate
cd $analysis_dir


# Check if the first argument ($2) is empty
if [ -z "$1" ]; then
  # If empty, assign a default value to a new variable
  ARG1=one_prosocial
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
  ARG3=persuasion
else
  # Otherwise, use the provided argument
  ARG3="$3"
fi

python3 examples/leader_sentiment.py $ARG1 $ARG2 $ARG3

