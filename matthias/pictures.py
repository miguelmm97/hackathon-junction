import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

# ----------------------------
# Settings
# ----------------------------
n_qubits = 8
wires = list(range(n_qubits))

# Identity matrices are only used as placeholders for drawing
I_global = np.eye(2**n_qubits, dtype=complex)
I_single = np.eye(2, dtype=complex)

# ----------------------------
# Custom labeled multi-qubit gate
# ----------------------------
class LabeledGlobalUnitary(qml.QubitUnitary):
    def __init__(self, matrix, wires, text):
        self._text = text
        super().__init__(matrix, wires=wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return self._text


# ----------------------------
# Custom unlabeled single-qubit gate
# ----------------------------
class BlankSingleQubitGate(qml.QubitUnitary):
    def label(self, decimals=None, base_label=None, cache=None):
        return ""


# ----------------------------
# Device
# ----------------------------
dev = qml.device("default.qubit", wires=n_qubits)


# ----------------------------
# Circuit
# ----------------------------
@qml.qnode(dev)
def circuit():
    # U1 and U1^\dagger
    LabeledGlobalUnitary(I_global, wires=wires, text=r"$U_1$")
    LabeledGlobalUnitary(I_global, wires=wires, text=r"$U_1^\dagger$")

    # Column of unlabeled 1-qubit gates
    for w in wires:
        BlankSingleQubitGate(I_single, wires=w)

    # U2 and U2^\dagger
    LabeledGlobalUnitary(I_global, wires=wires, text=r"$U_2$")
    LabeledGlobalUnitary(I_global, wires=wires, text=r"$U_2^\dagger$")

    # Another column of unlabeled 1-qubit gates
    for w in wires:
        BlankSingleQubitGate(I_single, wires=w)

    # U3 and U3^\dagger
    LabeledGlobalUnitary(I_global, wires=wires, text=r"$U_3$")
    LabeledGlobalUnitary(I_global, wires=wires, text=r"$U_3^\dagger$")

    # Final measurement
    return qml.sample(wires=wires)


# ----------------------------
# Draw and save
# ----------------------------
qml.drawer.use_style("black_white")  # try also: "pennylane", "sketch"

fig, ax = qml.draw_mpl(circuit)()

fig.set_size_inches(14, 8)
plt.tight_layout()
plt.savefig("quantum_circuit.png", dpi=300, bbox_inches="tight")
plt.show()