from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import math

QASM_PATH = "matthias/problem 5/challenge-32_5.qasm"
RIGHT = "011110010000101010001000"
RIGHT = "00111000101010100001000000010000"
SHOTS = 4096
SEED = 12345


def majority_vote(counts):
    total = sum(counts.values())
    n = len(next(iter(counts)))

    return "".join(
        "1" if sum(c for s, c in counts.items() if s[i] == "1") > total / 2 else "0"
        for i in range(n)
    )


def pairwise_refined_vote(counts, weak_threshold=0.05, edge_threshold=0.10, eps=1e-12):
    total = sum(counts.values())
    n = len(next(iter(counts)))

    # Start from ordinary majority vote.
    p1 = []
    result = []

    for i in range(n):
        ones = sum(c for s, c in counts.items() if s[i] == "1")
        p = ones / total
        p1.append(p)
        result.append("1" if p > 0.5 else "0")

    weak = [i for i, p in enumerate(p1) if abs(p - 0.5) <= weak_threshold]

    # Build strong same/different constraints between weak bits.
    edges = []
    pair_tables = {}

    for a, i in enumerate(weak):
        for j in weak[a + 1:]:
            table = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}

            for s, c in counts.items():
                xi, xj = int(s[i]), int(s[j])
                table[(xi, xj)] += c

            same = table[(0, 0)] + table[(1, 1)]
            diff = table[(0, 1)] + table[(1, 0)]
            strength = abs(same - diff) / total

            if strength >= edge_threshold:
                parity = 0 if same >= diff else 1
                edges.append((strength, i, j, parity))
                pair_tables[(i, j)] = table

    # Maximum spanning forest over weak bits.
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
            selected.append((i, j, parity))

    adj = {i: [] for i in weak}
    for i, j, parity in selected:
        adj[i].append((j, parity))
        adj[j].append((i, parity))

    # Fix each component up to a global flip.
    visited = set()

    for root in weak:
        if root in visited:
            continue

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

        if len(rel) == 1:
            continue

        scores = []

        for flip in (0, 1):
            score = 0.0

            for i in rel:
                xi = rel[i] ^ flip
                pi = p1[i] if xi else 1 - p1[i]
                score += math.log(pi + eps)

            for i, j, _ in selected:
                if i in rel and j in rel:
                    xi = rel[i] ^ flip
                    xj = rel[j] ^ flip
                    table = pair_tables[(min(i, j), max(i, j))]
                    if i > j:
                        xi, xj = xj, xi
                    score += math.log(table[(xi, xj)] / total + eps)

            scores.append(score)

        best_flip = 1 if scores[1] > scores[0] else 0

        for i in rel:
            result[i] = str(rel[i] ^ best_flip)

    return "".join(result)


qc = QuantumCircuit.from_qasm_file(QASM_PATH)
qc.measure_all()

print(f"target: {RIGHT}\n")

for bond_dim in range(2, 64):
    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=bond_dim,
        seed_simulator=SEED,
    )

    qc_t = transpile(qc, sim, seed_transpiler=SEED)
    counts = sim.run(qc_t, shots=SHOTS).result().get_counts()

    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top, top_count = ranked[0]

    maj = majority_vote(counts)
    pair = pairwise_refined_vote(counts)

    print(f"bond dim={bond_dim:2d}")
    print(f"  top sample: {top}  p={top_count / SHOTS:.4f}  {'OK' if top == RIGHT else 'WRONG'}")
    print(f"  majority:   {maj}  {'OK' if maj == RIGHT else 'WRONG'}")
    print(f"  pairwise:   {pair}  {'OK' if pair == RIGHT else 'WRONG'}")
    print()