#!/bin/bash

if [[ -n "${HACKATHON_PYTHON:-}" ]]; then
    export PATH="$(dirname "$HACKATHON_PYTHON"):$PATH"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate hackathon
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate hackathon
elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate hackathon
elif [[ -x "/users/flormatt/hackathon-prep/.venv/bin/python" ]]; then
    export PATH="/users/flormatt/hackathon-prep/.venv/bin:$PATH"
elif command -v module >/dev/null 2>&1; then
    module load cray-python/3.11.7
    if [[ -x "/users/flormatt/hackathon-prep/.venv/bin/python" ]]; then
        export PATH="/users/flormatt/hackathon-prep/.venv/bin:$PATH"
    fi
else
    echo "No hackathon Python environment found."
    echo "Set HACKATHON_PYTHON, install conda hackathon, or create /users/flormatt/hackathon-prep/.venv."
    exit 1
fi

export PYTHONPATH="/users/flormatt/hackathon-prep:${PYTHONPATH:-}"
