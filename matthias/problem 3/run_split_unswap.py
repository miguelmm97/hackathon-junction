import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matthias.cut_scan import run_split_unswap


summary = run_split_unswap(
    problem_number=3,
    center_lines=[75, 175, 325],
    max_bond=32,
    cutoff=1e-4,
    state_max_bond=32,
    state_cutoff=1e-4,
    max_its=5,
    n_jobs=3,
    consolidate=True,
    sabre_trials=200,
)

print("Estimated bitstring, q[0] first:", summary["bitstring_q0_first"])
print("Estimated bitstring, Qiskit order:", summary["bitstring_qiskit_order"])
print("Final MPS info:", summary["final_mps_info"])
print("Output directory:", summary["output_dir"])
