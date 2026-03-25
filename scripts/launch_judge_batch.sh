#!/bin/bash

# Launch a batch of jobs to do sentiment analysis.
#
# Usage:
#   sh launch_job_batch.sh
#
# This script will launch a batch of jobs, each with a different seed.
# The script will launch the jobs in the background, and will not wait for them
# to complete.
#
# To launch a single job, use the launch_judge.sh script directly.
#
# Example:
#   sh launch_job_batch.sh


# Population Settings
populations=("one_prosocial" "one_altruistic" "one_competitive" "one_individualistic")

# Sentiment type: cooperation, persuasion, svo.
sentiment="persuasion"

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
    sbatch --job-name=gse_${pop_name}_${i}_${disinfo:0:1} ./launch_judge.sh $population $disinfo $sentiment
  done
done


