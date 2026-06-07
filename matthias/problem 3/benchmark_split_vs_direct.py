import csv
from collections import Counter
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matthias.cut_scan.split_unswap import (
    majority_vote_counts,
    majority_vote_bitstrings,
    run_split_unswap,
    shannon_entropy_counts,
)


PROBLEM_NUMBER = 3
QASM_PATH = Path(__file__).resolve().parent / "challenge-24_3.qasm"
CENTER_LINES = [75, 175, 325]
BOND_DIMS = list(range(1, 33))
SHOTS = 256
CUTOFF = 1e-4
SEED = 12345
SPLIT_MAX_ITS = 1
SABRE_TRIALS = 20

OUTPUT_DIR = Path(__file__).resolve().parent / "split_vs_direct_benchmark"
CSV_PATH = OUTPUT_DIR / "entropy_vs_bond_dimension.csv"
FIGURE_PATH = OUTPUT_DIR / "entropy_vs_bond_dimension.png"


def direct_mps_counts(bond_dim):
    circuit = QuantumCircuit.from_qasm_file(str(QASM_PATH))
    circuit.measure_all()

    simulator = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=bond_dim,
        seed_simulator=SEED + bond_dim,
    )
    transpiled = transpile(circuit, simulator, seed_transpiler=SEED)
    result = simulator.run(transpiled, shots=SHOTS).result()
    return result.get_counts()


def write_rows(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows):
    xs = [row["bond_dim"] for row in rows]
    direct = [row["direct_entropy"] for row in rows]
    split = [row["split_entropy"] for row in rows]

    plt.figure(figsize=(6, 4), dpi=180)
    plt.plot(xs, direct, marker="o", label="direct MPS")
    plt.plot(xs, split, marker="o", label="split unswap + zip-up")
    plt.xlabel("bond dimension")
    plt.ylabel("Shannon entropy of sampled outcomes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_PATH)
    plt.close()


rows = []
for bond_dim in BOND_DIMS:
    print(f"bond_dim={bond_dim}")

    direct_counts = direct_mps_counts(bond_dim)
    direct_majority = majority_vote_counts(direct_counts)
    direct_entropy = shannon_entropy_counts(direct_counts)

    try:
        if bond_dim < 2:
            raise ValueError("Skipping split-unswap below bond dimension 2")

        split_summary = run_split_unswap(
            problem_number=PROBLEM_NUMBER,
            center_lines=CENTER_LINES,
            max_bond=bond_dim,
            cutoff=CUTOFF,
            state_max_bond=bond_dim,
            state_cutoff=CUTOFF,
            max_its=SPLIT_MAX_ITS,
            n_jobs=3,
            mps_shots=SHOTS,
            consolidate=True,
            sabre_trials=SABRE_TRIALS,
            output_dir=OUTPUT_DIR / f"split_bd_{bond_dim:02d}",
        )
        split_counts = Counter(split_summary["samples"])
        split_majority_q0_first = majority_vote_bitstrings(split_summary["samples"])
        split_majority = split_majority_q0_first[::-1]
        split_entropy = shannon_entropy_counts(split_counts)
        split_peak = max(split_counts, key=split_counts.get)[::-1]
        split_unique = len(split_counts)
        split_error = ""
    except Exception as exc:
        split_entropy = math.nan
        split_majority = ""
        split_peak = ""
        split_unique = 0
        split_error = repr(exc)

    row = {
        "bond_dim": bond_dim,
        "shots": SHOTS,
        "direct_entropy": direct_entropy,
        "split_entropy": split_entropy,
        "direct_majority_qiskit_order": direct_majority,
        "split_majority_qiskit_order": split_majority,
        "direct_peak_qiskit_order": max(direct_counts, key=direct_counts.get),
        "split_peak_qiskit_order": split_peak,
        "direct_unique": len(direct_counts),
        "split_unique": split_unique,
        "split_error": split_error,
        "split_max_its": SPLIT_MAX_ITS,
        "sabre_trials": SABRE_TRIALS,
    }
    rows.append(row)
    write_rows(rows)
    plot_rows(rows)

    print(
        f"  direct H={direct_entropy:.4f}, split H={split_entropy:.4f}, "
        f"direct majority={direct_majority}, split majority={split_majority}"
    )
    if split_error:
        print(f"  split failed: {split_error}")

print(f"CSV: {CSV_PATH}")
print(f"Figure: {FIGURE_PATH}")
