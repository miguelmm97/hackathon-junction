import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matthias.cut_scan import run_identity_window_scan

if __name__ == "__main__":
    run_identity_window_scan(
        problem_number=3,
        stride=5,
        window_size=40,
        max_bond=32,
        cutoff=1e-4,
        norm="trace_overlap",
        run_swap_optimization=False,
        swap_max_its=3,
    )
