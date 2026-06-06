#!/bin/bash
#SBATCH --job-name=p35-merge
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/users/flormatt/hackathon-prep/slurm_logs/problem35/merge_%j.out
#SBATCH --error=/users/flormatt/hackathon-prep/slurm_logs/problem35/merge_%j.err

set -euo pipefail

cd /users/flormatt/hackathon-prep

source "matthias/problem 35/activate_hackathon_env.sh"

python "matthias/problem 35/merge_problem35_results.py"
