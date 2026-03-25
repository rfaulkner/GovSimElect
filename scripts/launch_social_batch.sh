#!/bin/bash

# Launch a batch of jobs.
#
# Usage:
#   sh launch_social_batch.sh
#
# This script will launch a batch of jobs, each with a different seed.
# The number of seeds is specified by the N_SEEDS variable.
# The script will launch the jobs in the background, and will not wait for them
# to complete.
#
# To launch a single job, use the launch_job.sh script directly.
#
# Example:
#   sh launch_job_batch.sh

# Population Settings
populations=("balanced" "lean_altruistic" "lean_competitive" "one_prosocial" "one_competitive" "one_individualistic" "one_altruistic" "none")

# Disinfo Settings
disinfo=(true false)

# model_name="Qwen/Qwen1.5-110B-Chat-GPTQ-Int4"
# model_name="gpt/gpt-4.1-2025-04-14"
# model_name="gpt/gpt-4o-2024-05-13"
model_name="openrouter-google/gemini-2.5-flash"

# Launch 16 jobs
for disinfo in "${disinfo[@]}"; do
  for population in "${populations[@]}"; do
    IFS="_" read -r -a splits <<< "$population"
    if [[ ${#splits[@]} -gt 1 ]]; then
      p1=${splits[0]}
      p2=${splits[1]}
      pop_name="${p1:0:1}-${p2:0:3}"
      pop="${splits[0]}_${splits[1]}"
    else
      pop_name=${splits[-1]}
      pop_name=${pop_name:0:3}
      pop=$population
    fi
    sbatch --job-name=gsesoc_${pop_name}_${i}_${disinfo:0:1} ./launch_social.sh $population $disinfo $model_name
  done
done


