from math import ceil
from pathlib import Path
import time

from joblib import Parallel, delayed
from qiskit import QuantumCircuit

from .cut_scan import (
    get_bond_sizes,
    get_tn_info,
    identity_score,
    identity_window_mpo,
    problem_qasm_path,
    qasm_gate_line_numbers,
    selected_cut_indices,
    write_records_csv,
)


def split_into_batches(items, batch_size):
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def balanced_batches(items, n_batches):
    if not items:
        return []
    batch_size = ceil(len(items) / max(1, n_batches))
    return split_into_batches(items, batch_size)


def identity_window_record(
    circuit,
    gate_lines,
    problem_number,
    cut_index,
    stride,
    window_size,
    max_bond,
    cutoff,
    norm,
    run_swap_optimization,
    swap_max_its,
    swap_hows,
):
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

    return {
        "problem_number": problem_number,
        "qasm_line": gate_lines[cut_index],
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


def evaluate_identity_window_batch(
    problem_number,
    cut_indices,
    stride,
    window_size,
    max_bond,
    cutoff,
    norm="trace_overlap",
    run_swap_optimization=False,
    swap_max_its=3,
    swap_hows=("both", "left", "right"),
):
    qasm_path = problem_qasm_path(problem_number)
    circuit = QuantumCircuit.from_qasm_file(qasm_path)
    gate_lines = qasm_gate_line_numbers(qasm_path)

    return [
        identity_window_record(
            circuit,
            gate_lines,
            problem_number,
            cut_index,
            stride,
            window_size,
            max_bond,
            cutoff,
            norm,
            run_swap_optimization,
            swap_max_its,
            swap_hows,
        )
        for cut_index in cut_indices
    ]


def evaluate_identity_window_cut_indices(
    problem_number,
    cut_indices,
    stride,
    window_size,
    max_bond,
    cutoff,
    norm="trace_overlap",
    run_swap_optimization=False,
    swap_max_its=3,
    swap_hows=("both", "left", "right"),
    n_jobs=1,
    worker_batch_size=None,
):
    cut_indices = list(cut_indices)
    if n_jobs <= 1:
        batches = [cut_indices]
    elif worker_batch_size is None:
        batches = balanced_batches(cut_indices, n_jobs)
    else:
        batches = split_into_batches(cut_indices, worker_batch_size)

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_identity_window_batch)(
            problem_number,
            batch,
            stride,
            window_size,
            max_bond,
            cutoff,
            norm,
            run_swap_optimization,
            swap_max_its,
            swap_hows,
        )
        for batch in batches
    )

    records = [record for batch_records in results for record in batch_records]
    records.sort(key=lambda row: row["cut_index"])
    return records


def valid_identity_window_cut_indices(problem_number, window_size, stride):
    qasm_path = problem_qasm_path(problem_number)
    gate_lines = qasm_gate_line_numbers(qasm_path)
    return selected_cut_indices(len(gate_lines), window_size, stride)


def write_identity_window_records(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_records_csv(records, path)
