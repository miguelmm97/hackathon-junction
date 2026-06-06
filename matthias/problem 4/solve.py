from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

QASM_PATH = "matthias/problem 4/challenge-28_4.qasm"
SHOTS = 8192
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
