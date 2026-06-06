#!/bin/bash
set -euo pipefail

cd /users/flormatt/hackathon-prep

source "matthias/problem 35/activate_hackathon_env.sh"

mkdir -p "matthias/problem 35/identity_window_slurm/logs" "slurm_logs/problem35"

: "${LUMI_ACCOUNT:?Set LUMI_ACCOUNT to your project allocation, for example LUMI_ACCOUNT=project_...}"
LUMI_PARTITION="${LUMI_PARTITION:-standard}"
SBATCH_ARGS=(--account="$LUMI_ACCOUNT" --partition="$LUMI_PARTITION")

task_output=$(python "matthias/problem 35/make_problem35_tasks.py")
tasks_path=$(printf "%s\n" "$task_output" | sed -n "1p")
num_tasks=$(printf "%s\n" "$task_output" | sed -n "2p")
last_task=$((num_tasks - 1))

echo "Tasks: $num_tasks"
echo "Task table: $tasks_path"
echo "Account: $LUMI_ACCOUNT"
echo "Partition: $LUMI_PARTITION"

array_job=$(sbatch --parsable "${SBATCH_ARGS[@]}" --array=0-"$last_task" "matthias/problem 35/problem35_window_scan.sbatch")
echo "Array job: $array_job"

merge_job=$(sbatch --parsable "${SBATCH_ARGS[@]}" --dependency=afterok:"$array_job" "matthias/problem 35/problem35_merge.sbatch")
echo "Merge job: $merge_job"
