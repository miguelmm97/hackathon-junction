from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def build_hidden_parity_oracle_gate(secret_bits: str):
    """
    Build the oracle gate U_f for the Bernstein-Vazirani problem.

    This function represents the "problem setter" side of the problem.

    The solver should not inspect this function or read secret_bits.
    It only receives the resulting gate U_f.

    The oracle implements

        U_f |x>|y> = |x>|y xor f_s(x)>

    where

        f_s(x) = s · x mod 2.

    For secret_bits = "10110", the function is

        f_s(x) = x4 xor x2 xor x1.

    This is implemented by CNOTs into the output qubit:
        q4 -> output
        q2 -> output
        q1 -> output

    Multiple CNOTs into the same output qubit accumulate by xor,
    which is exactly addition modulo 2.
    """
    num_query_qubits = len(secret_bits)
    output_qubit = num_query_qubits

    oracle_circuit = QuantumCircuit(num_query_qubits + 1, name="U_f")

    # Qiskit qubit 0 corresponds to the rightmost printed bit.
    # Thus, for secret_bits = s4 s3 s2 s1 s0, we map:
    #
    #     s0 -> q0
    #     s1 -> q1
    #     ...
    #     s4 -> q4
    for query_qubit, secret_bit in enumerate(reversed(secret_bits)):
        if secret_bit == "1":
            oracle_circuit.cx(query_qubit, output_qubit)

    oracle_gate = oracle_circuit.to_gate(label="U_f")
    return oracle_gate, oracle_circuit


def build_bernstein_vazirani_solver_circuit(
    num_query_qubits: int,
    oracle_gate,
) -> QuantumCircuit:
    """
    Build the solver circuit.

    This is the part we are allowed to write.

    Input:
        num_query_qubits: the length n of the hidden string
        oracle_gate:      a black-box gate implementing U_f

    The solver does not know the hidden string.
    """
    output_qubit = num_query_qubits
    num_classical_bits = num_query_qubits

    query_qubits = list(range(num_query_qubits))

    circuit = QuantumCircuit(num_query_qubits + 1, num_classical_bits)

    # Prepare the output/ancilla qubit in |->.
    #
    # This is the phase-kickback trick:
    #
    #     X |0> = |1>
    #     H |1> = |-> = (|0> - |1>) / sqrt(2)
    circuit.x(output_qubit)
    circuit.h(output_qubit)

    # Prepare the query register in a uniform superposition:
    #
    #     |0...0> -> 1/sqrt(2^n) sum_x |x>
    circuit.h(query_qubits)

    # Black-box oracle call.
    #
    # The solver does not know how U_f is implemented internally.
    circuit.append(oracle_gate, query_qubits + [output_qubit])

    # Decode the phase pattern into the hidden string.
    circuit.h(query_qubits)

    # Measure only the query qubits.
    circuit.measure(query_qubits, range(num_classical_bits))

    return circuit


def make_hadamard_noise_model(hadamard_error_probability: float) -> NoiseModel:
    """
    Noise model where only H gates are noisy.

    In this artificial example, any loss of success probability is caused by
    depolarizing noise after Hadamard gates. The oracle CNOTs and measurements
    are still ideal.
    """
    noise_model = NoiseModel()

    if hadamard_error_probability > 0.0:
        hadamard_error = depolarizing_error(hadamard_error_probability, 1)
        noise_model.add_all_qubit_quantum_error(hadamard_error, ["h"])

    return noise_model


def run_circuit(
    circuit: QuantumCircuit,
    shots: int,
    hadamard_error_probability: float = 0.0,
) -> dict[str, int]:
    """Run the circuit on Aer, optionally with noise only on H gates."""
    noise_model = make_hadamard_noise_model(hadamard_error_probability)
    simulator = AerSimulator(noise_model=noise_model)

    compiled_circuit = transpile(circuit, simulator, optimization_level=0)
    result = simulator.run(compiled_circuit, shots=shots).result()

    return result.get_counts()


def main() -> None:
    # In the real black-box problem, the solver should not know this.
    # It is here only so that we can build a complete local demo.
    secret_bits = "10110"

    num_query_qubits = len(secret_bits)
    shots = 1000

    oracle_gate, oracle_circuit = build_hidden_parity_oracle_gate(secret_bits)

    solver_circuit = build_bernstein_vazirani_solver_circuit(
        num_query_qubits=num_query_qubits,
        oracle_gate=oracle_gate,
    )

    print(f"""
Bernstein-Vazirani demo
=======================

Hidden string, known only to the problem setter:
    s = {secret_bits}

The solver is given only:
    n = {num_query_qubits}
    black-box oracle gate U_f

Solver circuit, with the oracle shown as a black box:
""")

    print(solver_circuit.draw(output="text", fold=-1))

    print("""
For teaching only, here is the inside of the oracle.
The solver should not inspect this in the actual problem.
""")

    print(oracle_circuit.draw(output="text", fold=-1))

    print("""
Why the CNOT oracle works
=========================

The oracle should compute

    |x>|y> -> |x>|y xor f_s(x)>

with

    f_s(x) = s · x mod 2.

For s = 10110, this means

    f_s(x) = x4 xor x2 xor x1.

A CNOT from qi to the output qubit flips the output exactly when xi = 1.
Several such CNOTs into the same output qubit therefore compute the xor of
the selected bits. That is precisely s · x mod 2.

The Bernstein-Vazirani trick is to prepare the output qubit in |->.
Since X|-> = -|->, each controlled flip becomes a phase. The oracle therefore
turns

    |x>|-> -> (-1)^(s · x) |x>|->.

The final Hadamards convert that phase pattern back into |s>.
""")

    for hadamard_error_probability in [0.0, 0.01, 0.03, 0.05]:
        counts = run_circuit(
            circuit=solver_circuit,
            shots=shots,
            hadamard_error_probability=hadamard_error_probability,
        )

        success_probability = counts.get(secret_bits, 0) / shots
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

        print(f"""
Run result
==========

Hadamard depolarizing error probability:
    p_H = {hadamard_error_probability}

Success probability:
    P(measure s) = {success_probability:.3f}

Most common outcomes:
    {sorted_counts[:8]}
""")


if __name__ == "__main__":
    main()