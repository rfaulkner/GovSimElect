#!/bin/bash

# Launch a batch of jobs.
#
# Usage:
#   sh launch_job_batch.sh
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

# Number of seeds to launch.
N_SEEDS=4

# Population Settings
populations=("balanced" "none")

# Disinfo Settings
disinfo=(true false)


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
    for (( i=1; i<=$N_SEEDS; i++ )); do
      sbatch --job-name=gse_${pop_name}_${i}_${disinfo:0:1} ./launch_job.sh $i $population $disinfo
    done
  done
done
