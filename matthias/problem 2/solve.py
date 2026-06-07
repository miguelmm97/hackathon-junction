from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

QASM_PATH = "matthias/problem 2/challenge-16_2.qasm"
TOP_K = 10

qc = QuantumCircuit.from_qasm_file(QASM_PATH)
qc_no_meas = qc.remove_final_measurements(inplace=False)

plt.figure(figsize=(8, 4))
qc.draw(output='mpl', fold=0)
plt.savefig("qc.png", dpi=300)
plt.show()
##

sv = Statevector.from_instruction(qc_no_meas)
probs = sv.probabilities_dict()
ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)

plt.figure(figsize=(4, 4))
plt.plot(probs.values())
plt.show()

print(f"qasm: {QASM_PATH}")
print(f"qubits: {qc.num_qubits}")
print(f"gates: {sum(qc.count_ops().values())}")
print(f"gate counts: {dict(qc.count_ops())}")
print(f"depth: {qc.depth()}")
print(f"measurements: {'measure' in qc.count_ops()}")
print()
print("top candidates:")

for rank, (bitstring, probability) in enumerate(ranked[:TOP_K], start=1):
    print(f"{rank:2d}. {bitstring}  probability={probability:.12f}")

peak_bitstring, peak_prob = ranked[0]
print()
print(f"best bitstring: {peak_bitstring}")
print(f"peak probability: {peak_prob:.12f}")
