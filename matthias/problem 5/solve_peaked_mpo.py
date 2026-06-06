import csv
import sys
from collections import Counter
from pathlib import Path

from matplotlib import pyplot as plt
from qiskit import QuantumCircuit

PEAKED_PATH = Path("matthias/peaked-circuit-simulation")
sys.path.insert(0, str(PEAKED_PATH))

from unswap import mpo_compress_unswap, mpo_to_mps

QASM_PATH = "matthias/problem 5/challenge-32_5.qasm"
SHOTS = 1024
TOP_K = 10
SEED = 12345
MAX_BOND = 8192
CUTOFF = 0.002
UNSWAP_THRESHOLD = 1e6
CENTER_RATIO = 0.5
HOWS = ("both", "left", "right")
STATS_PATH = "matthias/problem 5/peaked_mpo_stats.csv"

circuit = QuantumCircuit.from_qasm_file(QASM_PATH)

mpo, layers_left, layers_right, stats_data = mpo_compress_unswap(
    circuit,
    max_bond=MAX_BOND,
    cutoff=CUTOFF,
    unswap_threshold=UNSWAP_THRESHOLD,
    center_ratio=CENTER_RATIO,
    early_stopping_gates=0,
    hows=HOWS,
    seed=SEED,
)

mps, perm = mpo_to_mps(
    mpo,
    layers_left[:-2],
    layers_right,
    max_bond=MAX_BOND,
    cutoff=CUTOFF,
)

raw_samples = [sample for sample, _ in list(mps.sample(SHOTS))]
samples = ["".join(str(bit) for bit in sample) for sample in raw_samples]
samples = ["".join(bitstring[i] for i in perm) for bitstring in samples]
counts = Counter(samples)
ranked = counts.most_common()

points = sorted(
    ((int(bitstring, 2), count) for bitstring, count in counts.items()),
    key=lambda item: item[0],
)

xs = [index for index, count in points]
ys = [count for index, count in points]
plt.figure(figsize=(4, 4), dpi=600)
plt.plot(xs, ys)
plt.show()

if stats_data:
    fieldnames = sorted({key for row in stats_data for key in row})
    with open(STATS_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_data)

print(f"qasm: {QASM_PATH}")
print(f"qubits: {circuit.num_qubits}")
print(f"gates: {sum(circuit.count_ops().values())}")
print(f"gate counts: {dict(circuit.count_ops())}")
print(f"depth: {circuit.depth()}")
print(f"shots: {SHOTS}")
print(f"max bond: {MAX_BOND}")
print(f"cutoff: {CUTOFF}")
print(f"center ratio: {CENTER_RATIO}")
print(f"unswap options: {HOWS}")
print(f"stats csv: {STATS_PATH}")
print()
##
print("top candidates:")

for rank, (bitstring, count) in enumerate(ranked[:TOP_K], start=1):
    print(f"{rank:2d}. {bitstring}  count={count}  probability={count / SHOTS:.12f}")

peak_bitstring, peak_count = ranked[0]
print()
print(f"estimated best bitstring: {peak_bitstring}")
print(f"estimated peak probability: {peak_count / SHOTS:.12f}")
