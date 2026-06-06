import csv

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

QASM_PATH = "matthias/problem 4/challenge-28_4.qasm"
REFERENCE_BITSTRING = None
SHOTS = 1024
BOND_DIMS = [2, 4, 8, 16, 32, 64]
SEEDS = [1001, 1002, 1003, 1004, 1005]
RESULTS_PATH = "matthias/problem 4/bond_dimension_sweep.csv"


def majority_vote_counts(counts):
    total = sum(counts.values())
    n = len(next(iter(counts)))

    return "".join(
        "1" if sum(c for s, c in counts.items() if s[i] == "1") > total / 2 else "0"
        for i in range(n)
    )


def pairwise_parity_vote(counts, max_edges=None):
    n = len(next(iter(counts)))
    total = sum(counts.values())
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            same = 0
            diff = 0
            for s, c in counts.items():
                if s[i] == s[j]:
                    same += c
                else:
                    diff += c
            parity = 0 if same >= diff else 1
            confidence = abs(same - diff) / total
            edges.append((confidence, i, j, parity))

    edges.sort(reverse=True)
    if max_edges is not None:
        edges = edges[:max_edges]

    parent = list(range(n))
    xor_to_parent = [0] * n

    def find(x):
        if parent[x] != x:
            root, xr = find(parent[x])
            xor_to_parent[x] ^= xr
            parent[x] = root
        return parent[x], xor_to_parent[x]

    def union(a, b, parity):
        ra, xa = find(a)
        rb, xb = find(b)
        if ra == rb:
            return
        parent[ra] = rb
        xor_to_parent[ra] = xa ^ xb ^ parity

    for _, i, j, parity in edges:
        union(i, j, parity)

    anchor = max(counts, key=counts.get)
    result = ["0"] * n
    for i in range(n):
        root, xr = find(i)
        result[i] = str(int(anchor[root]) ^ xr)

    return "".join(result)


def hamming_distance(left, right):
    return sum(a != b for a, b in zip(left, right))


qc_no_meas = QuantumCircuit.from_qasm_file(QASM_PATH)
qc = qc_no_meas.copy()
qc.measure_all()

reference_bitstring = REFERENCE_BITSTRING
if reference_bitstring is None:
    reference_probs = Statevector.from_instruction(qc_no_meas).probabilities()
    reference_index = reference_probs.argmax()
    reference_bitstring = format(reference_index, f"0{qc.num_qubits}b")

rows = []

for bond_dim in BOND_DIMS:
    for seed in SEEDS:
        sim = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=bond_dim,
            seed_simulator=seed,
        )

        qc_t = transpile(qc, sim, seed_transpiler=seed)
        result = sim.run(qc_t, shots=SHOTS).result()
        counts = result.get_counts()

        peak_bitstring = max(counts, key=counts.get)
        majority_bitstring = majority_vote_counts(counts)
        pairwise_bitstring = pairwise_parity_vote(counts)
        peak_count = counts[peak_bitstring]

        rows.append(
            {
                "bond_dim": bond_dim,
                "seed": seed,
                "shots": SHOTS,
                "peak_probability": peak_count / SHOTS,
                "unique_bitstrings": len(counts),
                "peak_bitstring": peak_bitstring,
                "majority_bitstring": majority_bitstring,
                "pairwise_bitstring": pairwise_bitstring,
                "peak_success": peak_bitstring == reference_bitstring,
                "majority_success": majority_bitstring == reference_bitstring,
                "pairwise_success": pairwise_bitstring == reference_bitstring,
                "peak_hamming": hamming_distance(peak_bitstring, reference_bitstring),
                "majority_hamming": hamming_distance(majority_bitstring, reference_bitstring),
                "pairwise_hamming": hamming_distance(pairwise_bitstring, reference_bitstring),
            }
        )

with open(RESULTS_PATH, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"reference bitstring: {reference_bitstring}")
print(f"shots per run: {SHOTS}")
print(f"seeds per bond dimension: {len(SEEDS)}")
print(f"results csv: {RESULTS_PATH}")
print()
print("bond_dim  peak_success  majority_success  pairwise_success  avg_peak_prob  avg_peak_hd  avg_majority_hd  avg_pairwise_hd")

for bond_dim in BOND_DIMS:
    subset = [row for row in rows if row["bond_dim"] == bond_dim]
    n = len(subset)
    peak_success = sum(row["peak_success"] for row in subset) / n
    majority_success = sum(row["majority_success"] for row in subset) / n
    pairwise_success = sum(row["pairwise_success"] for row in subset) / n
    avg_peak_prob = sum(row["peak_probability"] for row in subset) / n
    avg_peak_hd = sum(row["peak_hamming"] for row in subset) / n
    avg_majority_hd = sum(row["majority_hamming"] for row in subset) / n
    avg_pairwise_hd = sum(row["pairwise_hamming"] for row in subset) / n

    print(
        f"{bond_dim:8d}"
        f"  {peak_success:12.3f}"
        f"  {majority_success:16.3f}"
        f"  {pairwise_success:16.3f}"
        f"  {avg_peak_prob:13.3f}"
        f"  {avg_peak_hd:11.3f}"
        f"  {avg_majority_hd:15.3f}"
        f"  {avg_pairwise_hd:15.3f}"
    )
