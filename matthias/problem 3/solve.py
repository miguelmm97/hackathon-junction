from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

QASM_PATH = "matthias/problem 3/challenge-24_3.qasm"
SHOTS = 4096
BOND_DIM = 64
TOP_K = 10
SEED = 12345

qc = QuantumCircuit.from_qasm_file(QASM_PATH)
qc.measure_all()

sim = AerSimulator(
    method="matrix_product_state",
    matrix_product_state_max_bond_dimension=BOND_DIM,
    seed_simulator=SEED,
)

qc_t = transpile(qc, sim, seed_transpiler=SEED)
result = sim.run(qc_t, shots=SHOTS).result()
counts = result.get_counts()
ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)

plt.figure(figsize=(4, 4))
plt.plot(counts.values())
plt.show()

print(f"qasm: {QASM_PATH}")
print(f"qubits: {qc.num_qubits}")
print(f"gates: {sum(qc.count_ops().values())}")
print(f"gate counts: {dict(qc.count_ops())}")
print(f"depth: {qc.depth()}")
print(f"shots: {SHOTS}")
print(f"mps bond dimension: {BOND_DIM}")
print(f"seed: {SEED}")
print()
print("top candidates:")

for rank, (bitstring, count) in enumerate(ranked[:TOP_K], start=1):
    print(f"{rank:2d}. {bitstring}  count={count}  probability={count / SHOTS:.12f}")

peak_bitstring, peak_count = ranked[0]
print()
print(f"estimated best bitstring: {peak_bitstring}")
print(f"estimated peak probability: {peak_count / SHOTS:.12f}")

##

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

bitstrings = [bitstring for bitstring, probability in ranked]
majority_vote_counts(counts)
pairwise_parity_vote(counts)

