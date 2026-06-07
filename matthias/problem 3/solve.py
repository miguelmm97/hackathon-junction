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

import math


def majority_with_pairwise_weak_bits(
    counts,
    weak_threshold=0.05,
    edge_threshold=0.10,
    pair_weight=1.0,
    eps=1e-12,
):
    total = sum(counts.values())
    bitstrings = list(counts)
    n = len(bitstrings[0])

    # One-bit marginals and ordinary majority vote
    p1 = []
    majority = []

    for i in range(n):
        ones = sum(c for s, c in counts.items() if s[i] == "1")
        p = ones / total
        p1.append(p)
        majority.append("1" if p > 0.5 else "0")

    weak = [i for i, p in enumerate(p1) if abs(p - 0.5) <= weak_threshold]

    # Pairwise edges between weak bits
    edges = []
    pair_tables = {}

    for a, i in enumerate(weak):
        for j in weak[a + 1:]:
            table = {
                (0, 0): 0,
                (0, 1): 0,
                (1, 0): 0,
                (1, 1): 0,
            }

            for s, c in counts.items():
                xi = int(s[i])
                xj = int(s[j])
                table[(xi, xj)] += c

            same = table[(0, 0)] + table[(1, 1)]
            diff = table[(0, 1)] + table[(1, 0)]

            parity = 0 if same >= diff else 1
            strength = abs(same - diff) / total

            if strength >= edge_threshold:
                edges.append((strength, i, j, parity))
                pair_tables[(i, j)] = table
                pair_tables[(j, i)] = {
                    (b, a): v for (a, b), v in table.items()
                }

    # Maximum-spanning forest on weak-bit graph
    parent = {i: i for i in weak}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    selected = []
    for strength, i, j, parity in sorted(edges, reverse=True):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            selected.append((strength, i, j, parity))

    # Build adjacency of selected forest
    adj = {i: [] for i in weak}
    for strength, i, j, parity in selected:
        adj[i].append((j, parity))
        adj[j].append((i, parity))

    result = majority[:]
    visited = set()

    for root in weak:
        if root in visited:
            continue

        # Propagate relative bits in this component.
        stack = [(root, 0)]
        rel = {}
        visited.add(root)

        while stack:
            i, xi = stack.pop()
            rel[i] = xi

            for j, parity in adj[i]:
                if j not in visited:
                    visited.add(j)
                    stack.append((j, xi ^ parity))

        component = list(rel)

        # Compare the two possible orientations of this component.
        scores = []

        for flip in [0, 1]:
            score = 0.0

            # Unary contribution
            for i in component:
                xi = rel[i] ^ flip
                pi = p1[i] if xi == 1 else 1 - p1[i]
                score += math.log(pi + eps)

            # Pair contribution on selected edges inside component
            for _, i, j, _ in selected:
                if i in rel and j in rel:
                    xi = rel[i] ^ flip
                    xj = rel[j] ^ flip
                    table = pair_tables[(i, j)]
                    pij = table[(xi, xj)] / total
                    score += pair_weight * math.log(pij + eps)

            scores.append(score)

        best_flip = 1 if scores[1] > scores[0] else 0

        for i in component:
            result[i] = str(rel[i] ^ best_flip)

    return "".join(result), {
        "weak_bits": weak,
        "num_pair_edges": len(edges),
        "num_selected_edges": len(selected),
        "selected_edges": selected,
    }

candidate, info = majority_with_pairwise_weak_bits(
    counts,
    weak_threshold=0.05,
    edge_threshold=0.10,
)

print(candidate)
print(info["weak_bits"])
print(info["num_pair_edges"], info["num_selected_edges"])
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

