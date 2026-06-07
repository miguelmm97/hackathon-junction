import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from qiskit_quimb import quimb_circuit
from quimb.tensor import Circuit, CircuitMPS, MatrixProductOperator, tensor_network_1d_compress
from quimb.tensor.tn1d.compress import mps_gate_with_mpo_zipup

from .cut_scan import MATTHIAS_PATH, problem_qasm_path, qasm_gate_line_numbers, write_records_csv

PEAKED_PATH = MATTHIAS_PATH / "peaked-circuit-simulation"
sys.path.insert(0, str(PEAKED_PATH))

from circuit_mpo import apply_circuit, apply_mpo, mpo_from_circuit
from unswap import mpo_compress_unswap
from utils import get_tn_info, iter_layers, merge_layers, sample_tns

logging.getLogger().setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning, module="utils")


def executable_qasm_lines(qasm_path):
    gate_lines = set(qasm_gate_line_numbers(qasm_path))
    with open(qasm_path) as file:
        lines = file.readlines()
    header = [line for index, line in enumerate(lines, start=1) if index not in gate_lines]
    executable = [(index, line) for index, line in enumerate(lines, start=1) if index in gate_lines]
    return header, executable


def center_lines_to_slices(qasm_path, center_lines):
    _, executable = executable_qasm_lines(qasm_path)
    line_to_gate_index = {line_number: index for index, (line_number, _) in enumerate(executable)}
    missing_lines = [line_number for line_number in center_lines if line_number not in line_to_gate_index]
    if missing_lines:
        raise ValueError(f"Center lines are not executable QASM lines: {missing_lines}")

    center_indices = [line_to_gate_index[line_number] for line_number in center_lines]

    if center_indices != sorted(center_indices):
        raise ValueError("center_lines must be sorted in circuit order")

    boundaries = [
        (left + right + 1) // 2
        for left, right in zip(center_indices[:-1], center_indices[1:])
    ]
    starts = [0] + boundaries
    stops = boundaries + [len(executable)]

    return [
        {
            "part_index": index + 1,
            "start_index": start,
            "stop_index": stop,
            "center_line": center_lines[index],
            "center_index": center_indices[index],
            "local_center_index": center_indices[index] - start,
        }
        for index, (start, stop) in enumerate(zip(starts, stops))
    ]


def write_split_qasms(qasm_path, center_lines, output_dir=None, prefix=None):
    qasm_path = Path(qasm_path)
    header, executable = executable_qasm_lines(qasm_path)
    slices = center_lines_to_slices(qasm_path, center_lines)

    if output_dir is None:
        output_dir = qasm_path.parent / "split_unswap"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        prefix = qasm_path.stem

    for part in slices:
        part_path = output_dir / f"{prefix}.part{part['part_index']}.qasm"
        part_lines = [line for _, line in executable[part["start_index"] : part["stop_index"]]]
        with open(part_path, "w", newline="\n") as file:
            file.writelines(header)
            file.writelines(part_lines)
        part["qasm_path"] = str(part_path)

    return slices


def load_circuit(qasm_path, consolidate=False):
    circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
    circuit = circuit.remove_final_measurements(inplace=False)

    if consolidate:
        pass_manager = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)])
        circuit = pass_manager.run(circuit)

    return circuit


def qiskit_to_quimb_circuit(circuit):
    return quimb_circuit(circuit.decompose("unitary"), Circuit, to_backend=None)


def strip_measure_and_barrier_layers(layers):
    return [
        layer
        for layer in layers
        if "measure" not in layer.count_ops() and "barrier" not in layer.count_ops()
    ]


def final_measurement_perm(layers, num_qubits):
    measure_layers = [
        layer
        for layer in layers
        if "measure" in layer.count_ops() or "barrier" in layer.count_ops()
    ]
    if not measure_layers:
        return list(range(num_qubits))
    return [instruction.qubits[0]._index for instruction in measure_layers[-1]]


def swaps_for_output_permutation(perm):
    labels = list(range(len(perm)))
    swaps = []

    for target_pos, wanted_label in enumerate(perm):
        current_pos = labels.index(wanted_label)
        if current_pos != target_pos:
            swaps.append((target_pos, current_pos))
            labels[target_pos], labels[current_pos] = labels[current_pos], labels[target_pos]

    return swaps


def swap_circuit(num_qubits, swaps):
    circuit = QuantumCircuit(num_qubits)
    for q0, q1 in swaps:
        circuit.swap(q0, q1)
    return circuit


def compressed_result_to_segment_mpo(mpo_core, layers_left, layers_right, max_bond, cutoff):
    num_qubits = len(mpo_core.sites)
    segment_mpo = mpo_from_circuit(qiskit_to_quimb_circuit(QuantumCircuit(num_qubits)))

    left_layers = strip_measure_and_barrier_layers(layers_left)
    if left_layers:
        left_inverse = merge_layers(left_layers).inverse()
        for layer in iter_layers(left_inverse):
            segment_mpo = apply_circuit(
                segment_mpo,
                qiskit_to_quimb_circuit(layer),
                side="left",
                max_bond=max_bond,
                cutoff=cutoff,
            )

    segment_mpo = apply_mpo(
        segment_mpo,
        mpo_core,
        side="left",
        max_bond=max_bond,
        cutoff=cutoff,
    )

    for layer in strip_measure_and_barrier_layers(layers_right):
        segment_mpo = apply_circuit(
            segment_mpo,
            qiskit_to_quimb_circuit(layer),
            side="left",
            max_bond=max_bond,
            cutoff=cutoff,
        )

    swaps = swaps_for_output_permutation(final_measurement_perm(layers_right, num_qubits))
    if swaps:
        segment_mpo = apply_circuit(
            segment_mpo,
            qiskit_to_quimb_circuit(swap_circuit(num_qubits, swaps)),
            side="left",
            max_bond=max_bond,
            cutoff=cutoff,
        )

    return strict_1d_mpo(segment_mpo, max_bond=max_bond, cutoff=cutoff)


def strict_1d_mpo(mpo, max_bond, cutoff):
    compressed = tensor_network_1d_compress(
        mpo,
        max_bond=max_bond,
        cutoff=cutoff,
        method="direct",
        site_tags=[mpo.site_tag(site) for site in mpo.sites],
        optimize="greedy",
        inplace=False,
    )
    compressed.fuse_multibonds_()
    compressed_mpo = compressed.view_as_(MatrixProductOperator, cyclic=False, L=mpo.L)
    compressed_mpo._upper_ind_id = mpo._upper_ind_id
    compressed_mpo._lower_ind_id = mpo._lower_ind_id
    compressed_mpo._site_tag_id = mpo._site_tag_id
    compressed_mpo.ensure_bonds_exist()
    return compressed_mpo


def compress_qasm_part(
    part,
    max_bond,
    cutoff,
    unswap_threshold,
    early_stopping_gates,
    max_its,
    seed,
    consolidate,
    sabre_trials,
):
    started = time.perf_counter()
    circuit = load_circuit(part["qasm_path"], consolidate=consolidate)
    center_index = part["local_center_index"]
    if consolidate:
        center_index = min(center_index, len(circuit) // 2)

    mpo_core, layers_left, layers_right, stats = mpo_compress_unswap(
        circuit,
        max_bond=max_bond,
        cutoff=cutoff,
        unswap_threshold=unswap_threshold,
        early_stopping_gates=early_stopping_gates,
        center_ratio=center_index,
        equal=False,
        flip_freq=None,
        max_its=max_its,
        to_backend=None,
        seed=seed + part["part_index"],
        hows=("both", "left", "right"),
        sabre_trials=sabre_trials,
    )
    return {
        **part,
        "num_qubits": circuit.num_qubits,
        "num_ops": circuit.size(),
        "consolidate": consolidate,
        "sabre_trials": sabre_trials,
        "used_center_index": center_index,
        "stats_len": len(stats),
        "runtime_seconds": time.perf_counter() - started,
        "mpo_core": mpo_core,
        "layers_left": layers_left,
        "layers_right": layers_right,
        "mpo_info": get_tn_info(mpo_core),
    }


def zero_state_mps(num_qubits):
    return quimb_circuit(
        QuantumCircuit(num_qubits),
        quimb_circuit_class=CircuitMPS,
        to_backend=None,
    ).psi


def apply_mpo_zipup(mps, mpo, max_bond, cutoff):
    return mps_gate_with_mpo_zipup(
        mps,
        mpo,
        max_bond=max_bond,
        cutoff=cutoff,
        canonize=True,
        optimize="greedy",
    )


def apply_circuit_to_mps(mps, circuit, max_bond, cutoff):
    mpo = mpo_from_circuit(qiskit_to_quimb_circuit(circuit))
    return apply_mpo_zipup(mps, mpo, max_bond=max_bond, cutoff=cutoff)


def apply_compressed_segment_to_mps(mps, segment, state_max_bond, state_cutoff):
    num_qubits = segment["num_qubits"]

    left_layers = strip_measure_and_barrier_layers(segment["layers_left"])
    if left_layers:
        left_inverse = merge_layers(left_layers).inverse()
        for layer in iter_layers(left_inverse):
            mps = apply_circuit_to_mps(
                mps,
                layer,
                max_bond=state_max_bond,
                cutoff=state_cutoff,
            )

    mps = apply_mpo_zipup(
        mps,
        segment["mpo_core"],
        max_bond=state_max_bond,
        cutoff=state_cutoff,
    )

    for layer in strip_measure_and_barrier_layers(segment["layers_right"]):
        mps = apply_circuit_to_mps(
            mps,
            layer,
            max_bond=state_max_bond,
            cutoff=state_cutoff,
        )

    swaps = swaps_for_output_permutation(final_measurement_perm(segment["layers_right"], num_qubits))
    if swaps:
        mps = apply_circuit_to_mps(
            mps,
            swap_circuit(num_qubits, swaps),
            max_bond=state_max_bond,
            cutoff=state_cutoff,
        )

    return mps


def apply_compressed_segments_to_zero_state(segments, state_max_bond, state_cutoff):
    if not segments:
        raise ValueError("At least one compressed segment is required")

    mps = zero_state_mps(segments[0]["num_qubits"])
    records = []

    for segment in segments:
        started = time.perf_counter()
        mps = apply_compressed_segment_to_mps(
            mps,
            segment,
            state_max_bond=state_max_bond,
            state_cutoff=state_cutoff,
        )
        records.append(
            {
                "part_index": segment["part_index"],
                "runtime_seconds": time.perf_counter() - started,
                **get_tn_info(mps),
            }
        )

    return mps, records


def extract_marginal_bitstring(mps):
    projector_0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    bitstring = []
    p0s = []

    for site in mps.sites:
        try:
            p0 = mps.local_expectation_canonical(projector_0, where=[site], normalized=True)
        except ValueError:
            p0 = mps.local_expectation_exact(projector_0, where=[site], normalized=True)
        p0 = float(np.real(p0))
        p0s.append(p0)
        bitstring.append("0" if p0 >= 0.5 else "1")

    return "".join(bitstring), p0s


def sample_mps_fast(mps, num_samples, seed=123):
    samples = []
    sites = list(mps.sites)
    for config, _ in mps.sample(num_samples, seed=seed):
        samples.append("".join(str(config[site]) for site in sites))
    return samples


def majority_vote_bitstrings(bitstrings):
    total = len(bitstrings)
    if total == 0:
        raise ValueError("Cannot majority-vote an empty bitstring list")
    num_bits = len(bitstrings[0])

    return "".join(
        "1" if sum(bitstring[index] == "1" for bitstring in bitstrings) > total / 2 else "0"
        for index in range(num_bits)
    )


def majority_vote_counts(counts):
    bitstrings = []
    for bitstring, count in counts.items():
        bitstrings.extend([bitstring] * count)
    return majority_vote_bitstrings(bitstrings)


def shannon_entropy_counts(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * np.log2(probability)
    return float(entropy)


def run_split_unswap(
    problem_number,
    center_lines,
    max_bond=32,
    cutoff=1e-4,
    state_max_bond=None,
    state_cutoff=None,
    unswap_threshold=1e6,
    early_stopping_gates=0,
    max_its=5,
    seed=123,
    n_jobs=3,
    mps_shots=0,
    consolidate=False,
    sabre_trials=200,
    output_dir=None,
):
    qasm_path = problem_qasm_path(problem_number)
    if output_dir is None:
        output_dir = MATTHIAS_PATH / f"problem {problem_number}" / "split_unswap"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_max_bond = max_bond if state_max_bond is None else state_max_bond
    state_cutoff = cutoff if state_cutoff is None else state_cutoff

    parts = write_split_qasms(qasm_path, center_lines, output_dir=output_dir)
    results = Parallel(n_jobs=n_jobs)(
        delayed(compress_qasm_part)(
            part,
            max_bond,
            cutoff,
            unswap_threshold,
            early_stopping_gates,
            max_its,
            seed,
            consolidate,
            sabre_trials,
        )
        for part in parts
    )
    results.sort(key=lambda row: row["part_index"])

    mps, mps_records = apply_compressed_segments_to_zero_state(
        results,
        state_max_bond=state_max_bond,
        state_cutoff=state_cutoff,
    )
    bitstring_q0_first, p0s = extract_marginal_bitstring(mps)

    records = []
    for row in results:
        record = {
            key: value
            for key, value in row.items()
            if key not in ("mpo_core", "layers_left", "layers_right")
        }
        record["mpo_info"] = str(record["mpo_info"])
        records.append(record)

    write_records_csv(records, output_dir / "split_unswap_parts.csv")
    write_records_csv(mps_records, output_dir / "split_unswap_mps.csv")

    if mps_shots:
        try:
            samples = sample_mps_fast(mps, mps_shots, seed=seed)
        except Exception:
            samples = sample_tns(mps, mps_shots)
        majority_q0_first = majority_vote_bitstrings(samples)
    else:
        samples = []
        majority_q0_first = bitstring_q0_first

    summary = {
        "problem_number": problem_number,
        "center_lines": list(center_lines),
        "output_dir": str(output_dir),
        "bitstring_q0_first": majority_q0_first,
        "bitstring_qiskit_order": majority_q0_first[::-1],
        "marginal_bitstring_q0_first": bitstring_q0_first,
        "marginal_bitstring_qiskit_order": bitstring_q0_first[::-1],
        "p0s": p0s,
        "samples": samples,
        "part_records": records,
        "mps_records": mps_records,
        "final_mps_info": get_tn_info(mps),
    }
    return summary
