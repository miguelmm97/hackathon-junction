from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from matplotlib import pyplot as plt

QASM_PATH = "matthias/problem 5/challenge-32_5.qasm"
SHOTS = 1024
BOND_DIM = 64
TRUNCATION_THRESHOLD = 1e-4
TOP_K = 10

qc = QuantumCircuit.from_qasm_file(QASM_PATH)
qc.measure_all()

sim = AerSimulator(
    method="matrix_product_state",
    matrix_product_state_max_bond_dimension=BOND_DIM,
    matrix_product_state_truncation_threshold=TRUNCATION_THRESHOLD,
    mps_sample_measure_algorithm="mps_heuristic",
    device="CPU",
    mps_lapack=False,
    mps_omp_threads=1,
    max_parallel_threads=1,
    precision="double",
    seed_simulator=1,
)

qc_t = transpile(qc, sim, optimization_level=2)
result = sim.run(qc_t, shots=SHOTS).result()
counts = result.get_counts()
ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)

points = sorted(
    ((int(bitstring, 2), count) for bitstring, count in counts.items()),
    key=lambda item: item[0],
)

xs = [index for index, count in points]
ys = [count for index, count in points]
plt.figure(figsize=(4, 4), dpi=600)
plt.plot(xs, ys)
plt.show()


print(f"qasm: {QASM_PATH}")
print(f"qubits: {qc.num_qubits}")
print(f"gates: {sum(qc.count_ops().values())}")
print(f"gate counts: {dict(qc.count_ops())}")
print(f"depth: {qc.depth()}")
print(f"shots: {SHOTS}")
print(f"mps max bond dimension: {BOND_DIM}")
print(f"mps truncation threshold: {TRUNCATION_THRESHOLD}")
print()
print("top candidates:")

for rank, (bitstring, count) in enumerate(ranked[:TOP_K], start=1):
    print(f"{rank:2d}. {bitstring}  count={count}  probability={count / SHOTS:.12f}")

peak_bitstring, peak_count = ranked[0]
print()
print(f"estimated best bitstring: {peak_bitstring}")
print(f"estimated peak probability: {peak_count / SHOTS:.12f}")
