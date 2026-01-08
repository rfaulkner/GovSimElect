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

N_SEEDS=3

for (( i=1; i<=$N_SEEDS; i++ )); do
  sbatch ./launch_job.sh $i
done
