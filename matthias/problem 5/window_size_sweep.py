import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from joblib import Parallel, delayed

from matthias.cut_scan import run_identity_window_scan

WINDOW_SIZES = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100]
N_JOBS = 4

def run_window_size(window_size):
    return run_identity_window_scan(
        problem_number=5,
        stride=1,
        window_size=window_size,
        max_bond=32,
        cutoff=1e-4,
        norm="trace_overlap",
        run_swap_optimization=False,
        swap_max_its=3,
    )

if __name__ == "__main__":
    Parallel(n_jobs=N_JOBS)(
        delayed(run_window_size)(window_size)
        for window_size in WINDOW_SIZES
    )
