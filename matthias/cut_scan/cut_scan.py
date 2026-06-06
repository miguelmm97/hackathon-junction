import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import ElidePermutations, SabreSwap
from qiskit_quimb import quimb_circuit
from quimb.tensor import Circuit, MatrixProductOperator

MATTHIAS_PATH = Path(__file__).resolve().parents[1]
PEAKED_PATH = MATTHIAS_PATH / "peaked-circuit-simulation"
sys.path.insert(0, str(PEAKED_PATH))

from circuit_mpo import apply_circuit, apply_swaps, mpo_from_circuit
from unswap import unswap as optimize_mpo_swaps
from utils import elem_counts, get_tn_info, iter_layers, merge_gates, merge_layers


def problem_qasm_path(problem_number):
    problem_dir = MATTHIAS_PATH / f"problem {problem_number}"
    qasm_files = sorted(problem_dir.glob("*.qasm"))
    if len(qasm_files) != 1:
        raise ValueError(f"Expected one QASM file in {problem_dir}, found {len(qasm_files)}")
    return str(qasm_files[0])


def entropy_from_singular_values(values):
    weights = np.asarray(values, dtype=float) ** 2
    total = weights.sum()
    if total == 0:
        return 0.0
    weights = weights[weights > 0] / total
    return float(-(weights * np.log2(weights)).sum())


def required_bond_dimension(values, truncation_threshold, max_bond):
    weights = np.asarray(values, dtype=float) ** 2
    total = weights.sum()
    if total == 0:
        return 1

    weights = np.sort(weights / total)[::-1]
    discarded = 1.0 - np.cumsum(weights)
    required = len(weights)
    for index, tail_weight in enumerate(discarded, start=1):
        if tail_weight <= truncation_threshold:
            required = index
            break

    return min(required, max_bond)


def mpo_singular_values(mpo):
    values = []
    for bond in range(1, len(mpo.sites)):
        try:
            values.append(np.asarray(mpo.singular_values(bond), dtype=float))
        except Exception:
            values.append(np.array([], dtype=float))
    return values


def snapshot_mpo(
    records,
    spectra,
    mpo,
    center_ratio,
    step,
    stage,
    truncation_threshold,
    max_bond,
    elapsed,
    **extra,
):
    singular_values = mpo_singular_values(mpo)
    entropies = [entropy_from_singular_values(values) for values in singular_values]
    required_bonds = [
        required_bond_dimension(values, truncation_threshold, max_bond)
        for values in singular_values
    ]

    spectra[(center_ratio, step)] = singular_values
    records.append(
        {
            "center_ratio": center_ratio,
            "step": step,
            "stage": stage,
            "time": elapsed,
            "max_entropy": max(entropies, default=0.0),
            "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "max_required_bond": max(required_bonds, default=1),
            "mean_required_bond": float(np.mean(required_bonds)) if required_bonds else 1.0,
            "max_actual_bond": mpo.max_bond(),
            **get_tn_info(mpo),
            **extra,
        }
    )


def rewire_layers(layers, perm, seed=None):
    num_qubits = len(perm)
    qc = merge_layers(layers)
    qc = QuantumCircuit(num_qubits).compose(qc, qubits=np.argsort(perm))
    qc = ElidePermutations()(qc)

    swap_pass = SabreSwap(
        coupling_map=CouplingMap.from_line(layers[0].num_qubits),
        heuristic="decay",
        trials=10000,
        seed=seed,
    )
    qc = swap_pass(qc)
    return list(iter_layers(qc))


def get_bond_sizes(mpo):
    return np.array([mpo.bond_size(i, i + 1) for i in range(len(mpo.sites) - 1)])


def swap_perm(perm, swaps):
    for q0, q1 in swaps:
        perm[q0], perm[q1] = perm[q1], perm[q0]
    return perm


def get_good_swaps(mpo, qubit_pairs, how, max_bond, cutoff, equal=False):
    current_bonds = get_bond_sizes(mpo)
    swaps_l = qubit_pairs if how in ("left", "both") else []
    swaps_r = qubit_pairs if how in ("right", "both") else []

    mpo_tmp = apply_swaps(mpo, swaps_l=swaps_l, swaps_r=swaps_r, max_bond=max_bond, cutoff=cutoff)
    new_bonds = get_bond_sizes(mpo_tmp)

    if equal is None:
        new_bonds = new_bonds + (np.random.rand(*new_bonds.shape) - 0.5)
        return np.nonzero(new_bonds < current_bonds)[0]
    if equal:
        return np.nonzero(new_bonds <= current_bonds)[0]
    return np.nonzero(new_bonds < current_bonds)[0]


def instrumented_unswap(
    mpo,
    records,
    spectra,
    center_ratio,
    step,
    max_bond,
    cutoff,
    max_its,
    equal,
    hows,
    t0,
):
    num_qubits = len(mpo.sites)
    all_pairs = [(i, i + 1) for i in range(num_qubits - 1)]
    perm_left = list(range(num_qubits))
    perm_right = list(range(num_qubits))

    num_improvements = 1
    start_counts = 1
    end_counts = 0
    iteration = 0

    while num_improvements > 0 and iteration < max_its and start_counts != end_counts:
        num_improvements = 0
        start_counts = elem_counts(mpo)

        for how in hows:
            for parity in [0, 1]:
                swap_ids = get_good_swaps(
                    mpo,
                    qubit_pairs=all_pairs[parity::2],
                    how=how,
                    max_bond=max_bond,
                    cutoff=cutoff,
                    equal=equal,
                )
                swaps = [all_pairs[i] for i in swap_ids if i % 2 == parity]
                swaps_l = swaps if how in ("left", "both") else []
                swaps_r = swaps if how in ("right", "both") else []
                mpo = apply_swaps(mpo, swaps_l=swaps_l, swaps_r=swaps_r, max_bond=max_bond, cutoff=cutoff)

                if how in ("left", "both"):
                    perm_left = swap_perm(perm_left, swaps)
                if how in ("right", "both"):
                    perm_right = swap_perm(perm_right, swaps)

                num_improvements += len(swap_ids)
                step += 1
                snapshot_mpo(
                    records,
                    spectra,
                    mpo,
                    center_ratio,
                    step,
                    "unswapping",
                    cutoff,
                    max_bond,
                    time.perf_counter() - t0,
                    side=how,
                    parity=parity,
                    new_swaps=len(swap_ids),
                    total_swaps=num_improvements,
                )

        end_counts = elem_counts(mpo)
        iteration += 1

    return mpo, (perm_left, perm_right), step


def run_center_ratio(
    circuit,
    center_ratio,
    records,
    spectra,
    max_bond,
    cutoff,
    unswap_threshold,
    early_stopping_gates,
    equal,
    flip_freq,
    max_its,
    seed,
    hows,
):
    q2c = lambda qc: quimb_circuit(qc.decompose("unitary"), Circuit)
    t0 = time.perf_counter()

    split_index = int(len(circuit) * center_ratio) if isinstance(center_ratio, float) else center_ratio
    circuit_left = merge_gates(circuit[:split_index], circuit.num_qubits).inverse()
    circuit_right = merge_gates(circuit[split_index:], circuit.num_qubits)

    if "measure" not in circuit_left.count_ops():
        circuit_left.measure_all()
    if "measure" not in circuit_right.count_ops():
        circuit_right.measure_all()

    layers_left = rewire_layers(list(iter_layers(circuit_left)), np.arange(circuit.num_qubits), seed=seed)
    init_meas = layers_left[-2:]
    layers_left = layers_left[:-2]

    layers_right = rewire_layers(list(iter_layers(circuit_right)), np.arange(circuit.num_qubits), seed=seed)
    final_meas = layers_right[-2:]
    layers_right = layers_right[:-2]

    mpo = mpo_from_circuit(q2c(QuantumCircuit(circuit.num_qubits)))
    step = 0
    snapshot_mpo(records, spectra, mpo, center_ratio, step, "start", cutoff, max_bond, 0.0)

    ii_left = 0
    ii_right = 0
    do_left = False
    total_u_consumed = 0
    current_u_consumed = 0

    while ii_left < len(layers_left) or ii_right < len(layers_right):
        if ii_left < len(layers_left):
            mpo_left = apply_circuit(
                mpo,
                q2c(layers_left[ii_left].inverse()),
                side="right",
                max_bond=max_bond,
                cutoff=cutoff,
            )
            counts_left = elem_counts(mpo_left)
        else:
            mpo_left = None
            counts_left = 1e20

        if ii_right < len(layers_right):
            mpo_right = apply_circuit(
                mpo,
                q2c(layers_right[ii_right]),
                side="left",
                max_bond=max_bond,
                cutoff=cutoff,
            )
            counts_right = elem_counts(mpo_right)
        else:
            mpo_right = None
            counts_right = 1e20

        if flip_freq is None:
            do_left = counts_left < counts_right
        elif mpo_left is None:
            do_left = False
        elif mpo_right is None:
            do_left = True
        elif (ii_left + ii_right) % flip_freq == 0:
            do_left = not do_left

        if [counts_right, counts_left][int(do_left)] < unswap_threshold:
            if do_left:
                mpo = mpo_left
                ops = dict(layers_left[ii_left].count_ops())
                ii_left += 1
                side = "left"
            else:
                mpo = mpo_right
                ops = dict(layers_right[ii_right].count_ops())
                ii_right += 1
                side = "right"

            consumed = ops.get("unitary", 0)
            total_u_consumed += consumed
            current_u_consumed += consumed
            step += 1
            snapshot_mpo(
                records,
                spectra,
                mpo,
                center_ratio,
                step,
                "absorbing",
                cutoff,
                max_bond,
                time.perf_counter() - t0,
                absorb_side=side,
                it_left=ii_left,
                it_right=ii_right,
                u_consumed=consumed,
                u_consumed_total=total_u_consumed,
            )
        else:
            mpo, (perm_left, perm_right), step = instrumented_unswap(
                mpo,
                records,
                spectra,
                center_ratio,
                step,
                max_bond,
                cutoff,
                max_its,
                equal,
                hows,
                t0,
            )

            if ii_left < len(layers_left):
                layers_left = rewire_layers(layers_left[ii_left:] + init_meas, perm_left, seed=seed)
                init_meas = layers_left[-2:]
                layers_left = layers_left[:-2]
            else:
                layers_left = []

            if ii_right < len(layers_right):
                layers_right = rewire_layers(layers_right[ii_right:] + final_meas, perm_right, seed=seed)
                final_meas = layers_right[-2:]
                layers_right = layers_right[:-2]
            else:
                layers_right = []

            ii_left = 0
            ii_right = 0
            current_u_consumed = 0

            if len(circuit) - total_u_consumed <= early_stopping_gates:
                break

    return records, spectra


def write_spectra_npz(spectra, path):
    arrays = {}
    for (center_ratio, step), singular_values in spectra.items():
        label = str(center_ratio).replace(".", "p")
        for bond, values in enumerate(singular_values):
            arrays[f"cut_{label}_step_{step:04d}_bond_{bond:03d}"] = values
    np.savez_compressed(path, **arrays)


def write_records_csv(records, path):
    fieldnames = sorted({key for row in records for key in row})
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_records(records, output_dir):
    output_dir = Path(output_dir)

    for metric, ylabel, filename in [
        ("max_entropy", "max entanglement entropy", "cut_scan_entropy.png"),
        ("max_required_bond", "max required bond dimension", "cut_scan_required_bond.png"),
        ("max_actual_bond", "max actual bond dimension", "cut_scan_actual_bond.png"),
    ]:
        plt.figure(figsize=(7, 4), dpi=180)
        for center_ratio in sorted({row["center_ratio"] for row in records}):
            rows = [row for row in records if row["center_ratio"] == center_ratio]
            rows.sort(key=lambda row: row["step"])
            plt.plot(
                [row["step"] for row in rows],
                [row[metric] for row in rows],
                marker=".",
                linewidth=1.2,
                label=f"cut {center_ratio}",
            )

        plt.xlabel("algorithm step")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()


def run_cut_scan(
    problem_number,
    center_ratios,
    max_bond=64,
    cutoff=1e-4,
    unswap_threshold=1e6,
    early_stopping_gates=0,
    equal=False,
    flip_freq=None,
    max_its=20,
    seed=12345,
    hows=("both", "left", "right"),
    output_dir=None,
):
    qasm_path = problem_qasm_path(problem_number)
    circuit = QuantumCircuit.from_qasm_file(qasm_path)

    if output_dir is None:
        output_dir = MATTHIAS_PATH / f"problem {problem_number}" / "cut_scan"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    spectra = {}

    for center_ratio in center_ratios:
        print(f"running problem {problem_number}, center_ratio={center_ratio}")
        run_center_ratio(
            circuit,
            center_ratio,
            records,
            spectra,
            max_bond=max_bond,
            cutoff=cutoff,
            unswap_threshold=unswap_threshold,
            early_stopping_gates=early_stopping_gates,
            equal=equal,
            flip_freq=flip_freq,
            max_its=max_its,
            seed=seed,
            hows=hows,
        )

    write_records_csv(records, output_dir / "cut_scan_metrics.csv")
    write_spectra_npz(spectra, output_dir / "cut_scan_singular_values.npz")
    plot_records(records, output_dir)

    print(f"metrics csv: {output_dir / 'cut_scan_metrics.csv'}")
    print(f"singular values: {output_dir / 'cut_scan_singular_values.npz'}")
    print(f"plots: {output_dir}")
    return records, spectra


def qasm_gate_line_numbers(qasm_path):
    gate_lines = []
    ignored_prefixes = ("OPENQASM", "include", "qreg", "creg", "//")

    with open(qasm_path) as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line or line.startswith(ignored_prefixes):
                continue
            gate_lines.append(line_number)

    return gate_lines


def selected_cut_indices(num_gates, window_size, stride):
    first = window_size
    last = num_gates - window_size
    if first > last:
        raise ValueError("Window does not fit inside the circuit")
    if stride < 1:
        raise ValueError("stride must be at least 1")

    return list(range(first, last + 1, stride))


def qasm_lines_to_cut_indices(cut_lines, gate_lines, window_size):
    line_to_index = {line_number: index for index, line_number in enumerate(gate_lines)}
    cut_indices = []

    for line_number in cut_lines:
        if line_number not in line_to_index:
            raise ValueError(f"QASM line {line_number} is not an executable circuit line")
        cut_index = line_to_index[line_number]
        if cut_index < window_size or cut_index > len(gate_lines) - window_size:
            raise ValueError(f"QASM line {line_number} is too close to the circuit boundary")
        cut_indices.append(cut_index)

    return cut_indices


def identity_window_mpo(
    circuit,
    cut_index,
    window_size,
    max_bond,
    cutoff,
    run_swap_optimization=False,
    swap_max_its=3,
    swap_hows=("both", "left", "right"),
):
    q2c = lambda qc: quimb_circuit(qc.decompose("unitary"), Circuit)
    left = merge_gates(circuit[cut_index - window_size : cut_index], circuit.num_qubits).inverse()
    right = merge_gates(circuit[cut_index : cut_index + window_size], circuit.num_qubits)

    mpo = mpo_from_circuit(q2c(QuantumCircuit(circuit.num_qubits)))
    mpo = apply_circuit(mpo, q2c(left), side="right", max_bond=max_bond, cutoff=cutoff)
    mpo = apply_circuit(mpo, q2c(right), side="left", max_bond=max_bond, cutoff=cutoff)
    mpo = mpo.compress_all_1d(max_bond=max_bond, cutoff=cutoff)

    if run_swap_optimization:
        mpo, _, _ = optimize_mpo_swaps(
            mpo,
            hows=swap_hows,
            max_bond=max_bond,
            cutoff=cutoff,
            max_its=swap_max_its,
        )
        mpo = mpo.compress_all_1d(max_bond=max_bond, cutoff=cutoff)

    return mpo


def trace_identity_overlap(mpo, num_qubits):
    trace = complex(mpo.trace(optimize="greedy"))
    dimension = 2**num_qubits
    return min(1.0, abs(trace) / dimension)


def identity_score(mpo, num_qubits, norm):
    if norm == "trace_overlap":
        return trace_identity_overlap(mpo, num_qubits)
    raise ValueError(f"Unknown identity norm: {norm}")


def plot_identity_window_scan(records, output_dir):
    output_dir = Path(output_dir)
    window_size = int(records[0]["window_size"])
    swap_flag = int(bool(records[0]["run_swap_optimization"]))

    plt.figure(figsize=(7, 4), dpi=180)
    plt.plot(
        [row["qasm_line"] for row in records],
        [row["identity_overlap"] for row in records],
        marker=".",
        linewidth=1.2,
    )
    plt.title(f"W={window_size}, S={swap_flag}")
    plt.xlabel("QASM line at cut")
    plt.ylabel("identity overlap")
    plt.tight_layout()
    plt.savefig(output_dir / f"identity_window_overlap_W{window_size}_S{swap_flag}.png")
    plt.close()

    plt.figure(figsize=(7, 4), dpi=180)
    plt.plot(
        [row["qasm_line"] for row in records],
        [row["identity_distance"] for row in records],
        marker=".",
        linewidth=1.2,
    )
    plt.title(f"W={window_size}, S={swap_flag}")
    plt.xlabel("QASM line at cut")
    plt.ylabel("1 - identity overlap")
    plt.tight_layout()
    plt.savefig(output_dir / f"identity_window_distance_W{window_size}_S{swap_flag}.png")
    plt.close()


def run_identity_window_scan(
    problem_number,
    cut_lines=None,
    stride=5,
    window_size=40,
    max_bond=32,
    cutoff=1e-4,
    norm="trace_overlap",
    run_swap_optimization=False,
    swap_max_its=3,
    swap_hows=("both", "left", "right"),
    output_dir=None,
):
    qasm_path = problem_qasm_path(problem_number)
    circuit = QuantumCircuit.from_qasm_file(qasm_path)
    gate_lines = qasm_gate_line_numbers(qasm_path)

    if len(gate_lines) != len(circuit):
        raise ValueError(f"QASM gate lines ({len(gate_lines)}) do not match circuit instructions ({len(circuit)})")

    if cut_lines is None:
        cut_indices = selected_cut_indices(len(circuit), window_size, stride)
    else:
        cut_indices = qasm_lines_to_cut_indices(cut_lines, gate_lines, window_size)

    if output_dir is None:
        output_dir = MATTHIAS_PATH / f"problem {problem_number}" / "identity_window_scan"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for run_index, cut_index in enumerate(cut_indices, start=1):
        qasm_line = gate_lines[cut_index]
        print(f"{run_index:2d}/{len(cut_indices)}  qasm_line={qasm_line}  cut_index={cut_index}")

        started = time.perf_counter()
        mpo = identity_window_mpo(
            circuit,
            cut_index,
            window_size,
            max_bond,
            cutoff,
            run_swap_optimization=run_swap_optimization,
            swap_max_its=swap_max_its,
            swap_hows=swap_hows,
        )
        overlap = identity_score(mpo, circuit.num_qubits, norm)

        records.append(
            {
                "problem_number": problem_number,
                "qasm_line": qasm_line,
                "cut_index": cut_index,
                "stride": stride,
                "window_size": window_size,
                "max_bond": max_bond,
                "cutoff": cutoff,
                "norm": norm,
                "run_swap_optimization": run_swap_optimization,
                "swap_max_its": swap_max_its,
                "swap_hows": str(swap_hows),
                "identity_overlap": overlap,
                "identity_distance": 1.0 - overlap,
                "max_chain_bond": int(get_bond_sizes(mpo).max()),
                "max_internal_link": mpo.max_bond(),
                "runtime_seconds": time.perf_counter() - started,
                **get_tn_info(mpo),
            }
        )

    swap_flag = int(bool(run_swap_optimization))
    write_records_csv(records, output_dir / f"identity_window_metrics_W{window_size}_S{swap_flag}.csv")
    plot_identity_window_scan(records, output_dir)

    print(f"metrics csv: {output_dir / f'identity_window_metrics_W{window_size}_S{swap_flag}.csv'}")
    print(f"plots: {output_dir}")
    return records
