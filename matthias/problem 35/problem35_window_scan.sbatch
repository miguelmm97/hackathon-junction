#!/bin/bash
#SBATCH --job-name=p35-window
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/users/flormatt/hackathon-prep/slurm_logs/problem35/window_%A_%a.out
#SBATCH --error=/users/flormatt/hackathon-prep/slurm_logs/problem35/window_%A_%a.err

set -euo pipefail

cd /users/flormatt/hackathon-prep

source "matthias/problem 35/activate_hackathon_env.sh"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "matthias/problem 35/run_problem35_task.py"
