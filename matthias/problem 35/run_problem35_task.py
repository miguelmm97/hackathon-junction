from csv import DictReader
import os
from pathlib import Path

from matthias.cut_scan.cluster_window_scan import (
    evaluate_identity_window_cut_indices,
    valid_identity_window_cut_indices,
    write_identity_window_records,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "identity_window_slurm"
TASKS_PATH = OUTPUT_DIR / "tasks.csv"
PARTS_DIR = OUTPUT_DIR / "parts"

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
WORKER_BATCH_SIZE = int(os.environ.get("WINDOW_WORKER_BATCH_SIZE", "16"))


def load_task(task_id):
    with open(TASKS_PATH, newline="") as file:
        for row in DictReader(file):
            if int(row["task_id"]) == task_id:
                return row
    raise ValueError(f"Task id {task_id} not found in {TASKS_PATH}")


def main():
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    task = load_task(task_id)

    problem_number = int(task["problem_number"])
    window_size = int(task["window_size"])
    stride = int(task["stride"])
    max_bond = int(task["max_bond"])
    cutoff = float(task["cutoff"])
    start = int(task["start"])
    stop = int(task["stop"])

    cut_indices = valid_identity_window_cut_indices(problem_number, window_size, stride)[start:stop]
    records = evaluate_identity_window_cut_indices(
        problem_number=problem_number,
        cut_indices=cut_indices,
        stride=stride,
        window_size=window_size,
        max_bond=max_bond,
        cutoff=cutoff,
        norm="trace_overlap",
        run_swap_optimization=False,
        n_jobs=N_JOBS,
        worker_batch_size=WORKER_BATCH_SIZE,
    )

    part_path = PARTS_DIR / f"W{window_size:03d}_chunk{int(task['chunk_id']):04d}.csv"
    write_identity_window_records(records, part_path)
    print(part_path)


if __name__ == "__main__":
    main()
