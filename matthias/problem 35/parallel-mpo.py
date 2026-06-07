#!/usr/bin/env python3
"""Run MPO compression on three split QASM files in parallel, then combine them.

This script is meant to be used after ``split-qasm.py``. For example:

    python split-qasm.py qasm/challenge-8_1.qasm 0.33 0.66 \
        --output-dir split_qasm \
        --prefix split-challenge-8_1

    python parallel-mpo-main.py \
        split_qasm/split-challenge-8_1.part1.qasm \
        split_qasm/split-challenge-8_1.part2.qasm \
        split_qasm/split-challenge-8_1.part3.qasm \
        --max-bond 64 \
        --cutoff 0.01 \
        --mps-shots 1000

The three worker processes each run the same MPO attack/compression algorithm
used by ``run-mpo-qasm.py``. Each worker returns a full Matrix Product Operator
for its circuit segment. The parent process then composes the three segment
MPOs in circuit order:

    final_mpo = part3_mpo @ part2_mpo @ part1_mpo

Finally, it applies ``final_mpo`` to |0...0> and reads a bitstring either from
one-qubit marginals or from samples of the final MPS.

Note on Aer: Qiskit Aer cannot run a Quimb MPO object directly. The optional
``--aer-shots`` path runs Aer on the composed Qiskit circuit for comparison.
"""

import argparse
import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks


ROOT = Path(__file__).resolve().parent
MPO_DIR = ROOT / "MPO-attack-greedy"

sys.path.insert(0, str(MPO_DIR))
os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")
warnings.filterwarnings("ignore", category=FutureWarning, module="quimb")
warnings.filterwarnings("ignore", category=FutureWarning, module="utils")
warnings.filterwarnings(
    "ignore",
    message=r"Couldn't import `kahypar`.*",
    category=UserWarning,
    module=r"cotengra\..*"
           r""
           r""
           r"",
)

from circuit_mpo import apply_circuit, apply_mpo, mpo_from_circuit  # noqa: E402
from qiskit_quimb import quimb_circuit  # noqa: E402
from quimb.tensor import (  # noqa: E402
    Circuit,
    CircuitMPS,
    MatrixProductOperator,
    tensor_network_1d_compress,
)
from quimb.tensor.tn1d.compress import mps_gate_with_mpo_zipup  # noqa: E402
from unswap import mpo_compress_unswap  # noqa: E402
from utils import get_tn_info, iter_layers, merge_layers, sample_tns  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress three split QASM files in parallel, compose their MPOs, and read out the result."
    )
    parser.add_argument("part1", type=Path, help="First split QASM file.")
    parser.add_argument("part2", type=Path, help="Second split QASM file.")
    parser.add_argument("part3", type=Path, help="Third split QASM file.")
    parser.add_argument("--max-bond", type=int, default=256)
    parser.add_argument("--cutoff", type=float, default=0.002)
    parser.add_argument(
        "--state-max-bond",
        type=int,
        default=None,
        help="Max bond used when applying segment MPOs to the running MPS. Defaults to --max-bond.",
    )
    parser.add_argument(
        "--state-cutoff",
        type=float,
        default=None,
        help="Cutoff used when applying segment MPOs to the running MPS. Defaults to --cutoff.",
    )
    parser.add_argument("--unswap-threshold", type=float, default=1e6)
    parser.add_argument("--early-stopping-gates", type=int, default=0)
    parser.add_argument("--center-ratio", type=float, default=0.5)
    parser.add_argument("--max-its", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--flip-freq", type=int, default=None)
    parser.add_argument("--no-consolidate", action="store_true")
    parser.add_argument(
        "--mps-shots",
        type=int,
        default=0,
        help="Sample this many bitstrings from the final MPS produced by the combined MPO.",
    )
    parser.add_argument(
        "--aer-shots",
        type=int,
        default=0,
        help="Optional comparison: run this many Aer shots on the composed Qiskit circuit.",
    )
    parser.add_argument(
        "--aer-method",
        default="matrix_product_state",
        help="AerSimulator method for --aer-shots. Default: matrix_product_state.",
    )
    parser.add_argument(
        "--executor",
        choices=("process", "thread"),
        default="process",
        help="Parallel executor. Use process on a cluster; thread is useful where process pools are restricted.",
    )
    return parser.parse_args()


def load_circuit(qasm_path, consolidate=True):
    """Load a QASM file and optionally consolidate blocks into unitary ops."""
    circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
    circuit = circuit.remove_final_measurements(inplace=False)

    if consolidate:
        collect_2q = PassManager(
            [Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]
        )
        circuit = collect_2q.run(circuit)

    return circuit


def strip_measure_and_barrier_layers(layers):
    """Drop synthetic measure/barrier layers from MPO leftover layers."""
    return [
        layer
        for layer in layers
        if "measure" not in layer.count_ops() and "barrier" not in layer.count_ops()
    ]


def final_measurement_perm(layers, num_qubits):
    """Extract the measurement permutation inserted by ``mpo_compress_unswap``.

    ``mpo_to_mps`` uses this same information to reorder the final bitstring.
    When composing segment MPOs, we need to apply this permutation as real swap
    gates so that each segment MPO outputs qubits in logical q[0]..q[n-1] order
    before the next segment is applied.
    """
    measure_layers = [
        layer
        for layer in layers
        if "measure" in layer.count_ops() or "barrier" in layer.count_ops()
    ]
    if not measure_layers:
        return list(range(num_qubits))

    measurement_layer = measure_layers[-1]
    return [instruction.qubits[0]._index for instruction in measurement_layer]


def swaps_for_output_permutation(perm):
    """Return swaps that transform raw site order into logical output order.

    If ``raw`` is a raw MPS bitstring, ``mpo_to_mps`` reports the logical
    bitstring as ``raw[perm[0]] raw[perm[1]] ...``. Applying the swaps generated
    here to the raw state makes the tensor-network site order match that same
    logical order.
    """
    labels = list(range(len(perm)))
    swaps = []

    for target_pos, wanted_label in enumerate(perm):
        current_pos = labels.index(wanted_label)
        if current_pos == target_pos:
            continue
        swaps.append((target_pos, current_pos))
        labels[target_pos], labels[current_pos] = labels[current_pos], labels[target_pos]

    return swaps


def swap_circuit(num_qubits, swaps):
    """Build a Qiskit circuit that applies a list of swaps in order."""
    circuit = QuantumCircuit(num_qubits)
    for q0, q1 in swaps:
        circuit.swap(q0, q1)
    return circuit


def qiskit_to_quimb_circuit(qc, to_backend=None):
    """Convert a Qiskit circuit layer into the Quimb Circuit type expected here."""
    return quimb_circuit(qc.decompose("unitary"), Circuit, to_backend=to_backend)


def compressed_result_to_full_mpo(
    mpo_core,
    layers_left,
    layers_right,
    max_bond,
    cutoff,
    to_backend=None,
):
    """Turn ``mpo_compress_unswap`` output into one full segment MPO.

    ``mpo_compress_unswap`` returns only the central compressed MPO plus any
    layers that were not absorbed. ``mpo_to_mps`` applies those pieces to a
    state in this order:

        inverse(leftover_left_layers), then mpo_core, then leftover_right_layers

    This function applies the same order, but to an identity MPO, so the output
    is a standalone operator for the whole segment.
    """
    num_qubits = len(mpo_core.sites)

    identity = mpo_from_circuit(
        qiskit_to_quimb_circuit(QuantumCircuit(num_qubits), to_backend=to_backend)
    )
    full_mpo = identity

    left_layers = strip_measure_and_barrier_layers(layers_left)
    if left_layers:
        left_inverse = merge_layers(left_layers).inverse()
        for layer in iter_layers(left_inverse):
            full_mpo = apply_circuit(
                full_mpo,
                qiskit_to_quimb_circuit(layer, to_backend=to_backend),
                side="left",
                max_bond=max_bond,
                cutoff=cutoff,
            )

    full_mpo = apply_mpo(
        full_mpo,
        mpo_core,
        side="left",
        max_bond=max_bond,
        cutoff=cutoff,
    )

    for layer in strip_measure_and_barrier_layers(layers_right):
        full_mpo = apply_circuit(
            full_mpo,
            qiskit_to_quimb_circuit(layer, to_backend=to_backend),
            side="left",
            max_bond=max_bond,
            cutoff=cutoff,
        )

    perm = final_measurement_perm(layers_right, num_qubits)
    swaps = swaps_for_output_permutation(perm)
    if swaps:
        full_mpo = apply_circuit(
            full_mpo,
            qiskit_to_quimb_circuit(swap_circuit(num_qubits, swaps), to_backend=to_backend),
            side="left",
            max_bond=max_bond,
            cutoff=cutoff,
        )

    return full_mpo


def compress_one_part(index, qasm_path, args_dict):
    """Worker function: compress one split QASM file into a full MPO."""
    warnings.filterwarnings("ignore", category=UserWarning, module="autoray")

    circuit = load_circuit(qasm_path, consolidate=not args_dict["no_consolidate"])
    t0 = time.perf_counter()

    mpo_core, layers_left, layers_right, stats = mpo_compress_unswap(
        circuit,
        max_bond=args_dict["max_bond"],
        cutoff=args_dict["cutoff"],
        unswap_threshold=args_dict["unswap_threshold"],
        early_stopping_gates=args_dict["early_stopping_gates"],
        center_ratio=args_dict["center_ratio"],
        equal=False,
        flip_freq=args_dict["flip_freq"],
        max_its=args_dict["max_its"],
        to_backend=None,
        seed=args_dict["seed"] + index,
        hows=("both", "left", "right"),
    )

    full_mpo = compressed_result_to_full_mpo(
        mpo_core,
        layers_left,
        layers_right,
        max_bond=args_dict["max_bond"],
        cutoff=args_dict["cutoff"],
        to_backend=None,
    )

    return {
        "index": index,
        "qasm_path": str(qasm_path),
        "num_qubits": circuit.num_qubits,
        "num_ops": circuit.size(),
        "elapsed": time.perf_counter() - t0,
        "stats_len": len(stats),
        "mpo": full_mpo,
        "mpo_info": get_tn_info(full_mpo),
    }


def combine_segment_mpos(segment_mpos, max_bond, cutoff):
    """Compose segment MPOs in circuit order.

    If the three split circuits are U1, U2, U3, the final circuit applies:

        U3 * U2 * U1 * |0>

    ``compose_mpos_sitewise`` composes the next segment on the left of the
    current MPO.
    """
    combined = segment_mpos[0]
    for segment_mpo in segment_mpos[1:]:
        combined = compose_mpos_sitewise(
            combined,
            segment_mpo,
            max_bond=max_bond,
            cutoff=cutoff,
        )
    return combined


def mpo_site_array_lrud(mpo, site):
    """Return one MPO site as a rank-4 ``(left, right, upper, lower)`` array."""
    tensor = mpo[mpo.site_tag(site)]
    site_pos = mpo.sites.index(site)

    left_ind = mpo.bond(mpo.sites[site_pos - 1], site) if site_pos > 0 else None
    right_ind = (
        mpo.bond(site, mpo.sites[site_pos + 1])
        if site_pos < len(mpo.sites) - 1
        else None
    )
    upper_ind = mpo.upper_ind(site)
    lower_ind = mpo.lower_ind(site)

    existing_inds = [
        ind for ind in (left_ind, right_ind, upper_ind, lower_ind) if ind is not None
    ]
    data = tensor.transpose(*existing_inds).data

    if left_ind is None:
        data = np.expand_dims(data, axis=0)
    if right_ind is None:
        data = np.expand_dims(data, axis=1)

    return data


def drop_boundary_bonds_lrud(array, site_pos, num_sites):
    """Remove singleton boundary bond axes for ``MatrixProductOperator``."""
    if num_sites == 1:
        return array[0, 0, :, :]
    if site_pos == 0:
        return array[0, :, :, :]
    if site_pos == num_sites - 1:
        return array[:, 0, :, :]
    return array


def strict_1d_mpo(mpo, max_bond, cutoff):
    """Compress an MPO-like TN into a nearest-neighbor open-chain MPO."""
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
    compressed_mpo = compressed.view_as_(
        MatrixProductOperator,
        cyclic=False,
        L=mpo.L,
    )
    compressed_mpo._upper_ind_id = mpo._upper_ind_id
    compressed_mpo._lower_ind_id = mpo._lower_ind_id
    compressed_mpo._site_tag_id = mpo._site_tag_id
    compressed_mpo.ensure_bonds_exist()
    return compressed_mpo


def compose_mpos_sitewise(mpo_before, mpo_after, max_bond, cutoff):
    """Compose two MPOs as ``mpo_after @ mpo_before``.

    The per-site contraction sums over the physical leg connecting the two
    operators, while each new virtual leg is the product of the two input
    virtual legs. The reconstructed MPO is then SVD-compressed along the
    one-dimensional virtual chain.
    """
    mpo_before = strict_1d_mpo(mpo_before, max_bond=max_bond, cutoff=cutoff)
    mpo_after = strict_1d_mpo(mpo_after, max_bond=max_bond, cutoff=cutoff)

    if mpo_before.sites != mpo_after.sites:
        raise ValueError("MPO site labels must match before sitewise composition.")

    sites = tuple(mpo_before.sites)
    arrays = []
    for site_pos, site in enumerate(sites):
        before = mpo_site_array_lrud(mpo_before, site)
        after = mpo_site_array_lrud(mpo_after, site)

        if before.shape[2] != after.shape[3]:
            raise ValueError(
                "MPO physical dimensions are incompatible for composition at "
                f"site {site}: {before.shape} then {after.shape}"
            )

        composed = np.einsum("lrum,LRmd->lLrRud", after, before)
        composed = composed.reshape(
            after.shape[0] * before.shape[0],
            after.shape[1] * before.shape[1],
            after.shape[2],
            before.shape[3],
        )
        arrays.append(drop_boundary_bonds_lrud(composed, site_pos, len(sites)))

    composed_mpo = MatrixProductOperator(
        arrays,
        sites=sites,
        L=mpo_before.L,
        shape="lrud",
        upper_ind_id=mpo_before._upper_ind_id,
        lower_ind_id=mpo_before._lower_ind_id,
        site_tag_id=mpo_before._site_tag_id,
    )
    composed_mpo.compress_all_1d_(max_bond=max_bond, cutoff=cutoff)
    composed_mpo.ensure_bonds_exist()
    return composed_mpo


def zero_state_mps(num_qubits):
    """Return the |0...0> product state as an MPS."""
    mps = quimb_circuit(
        QuantumCircuit(num_qubits),
        quimb_circuit_class=CircuitMPS,
        to_backend=None,
    ).psi

    return mps


def apply_mpo_to_state(mpo, mps, max_bond, cutoff):
    """Apply one segment MPO to an MPS using Stoudenmire's zip-up algorithm."""
    return mps_gate_with_mpo_zipup(
        mps,
        mpo,
        max_bond=max_bond,
        cutoff=cutoff,
        canonize=True,
        optimize="greedy",
    )


def apply_segment_mpos_to_zero_state(segment_mpos, mpo_max_bond, mpo_cutoff, state_max_bond, state_cutoff):
    """Apply segment MPOs to |0...0> in circuit order without composing MPOs."""
    if not segment_mpos:
        raise ValueError("At least one segment MPO is required.")

    final_mps = zero_state_mps(len(segment_mpos[0].sites))
    print("Initial MPS info:", get_tn_info(final_mps))

    for index, segment_mpo in enumerate(segment_mpos, start=1):
        print(f"Normalizing segment MPO {index} to a strict 1D chain...")
        segment_mpo = strict_1d_mpo(
            segment_mpo,
            max_bond=mpo_max_bond,
            cutoff=mpo_cutoff,
        )
        print(f"Strict segment MPO {index} info:", get_tn_info(segment_mpo))

        print(f"Applying segment MPO {index} to current MPS with zip-up...")
        final_mps = apply_mpo_to_state(
            segment_mpo,
            final_mps,
            max_bond=state_max_bond,
            cutoff=state_cutoff,
        )
        print(f"After segment {index} MPS info:", get_tn_info(final_mps))

    return final_mps


def extract_peak_bitstring(mps):
    """Read a bitstring from one-qubit marginal probabilities."""
    pi0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    bitstring = []
    p0s = []

    for site in mps.sites:
        try:
            p0 = mps.local_expectation_canonical(
                pi0,
                where=[site],
                normalized=True,
            ).real.item()
        except ValueError:
            p0 = mps.local_expectation_exact(
                pi0,
                where=[site],
                normalized=True,
            ).real.item()
        p0s.append(p0)
        bitstring.append("1" if p0 < 0.5 else "0")

    return "".join(bitstring), p0s


def print_sample_summary(samples, predicted_bitstring):
    """Print empirical MPS shot counts next to the marginal prediction."""
    counts = Counter(samples)
    ranked = counts.most_common()
    predicted_count = counts.get(predicted_bitstring, 0)
    predicted_rank = next(
        (
            rank
            for rank, (bitstring, _) in enumerate(ranked, start=1)
            if bitstring == predicted_bitstring
        ),
        None,
    )

    print(
        "Marginal-predicted string MPS-shot count "
        f"({predicted_bitstring}, q[0]..q[n-1]): {predicted_count}/{len(samples)}"
    )
    print(
        "Marginal-predicted string MPS-shot rank: "
        f"{predicted_rank if predicted_rank is not None else 'not sampled'}"
    )

    for label, item in [
        ("Most frequent MPS-shot string", ranked[0] if ranked else (None, 0)),
        ("Second most frequent MPS-shot string", ranked[1] if len(ranked) > 1 else (None, 0)),
    ]:
        if item[0] is None:
            print(f"{label}: none, 0/{len(samples)}")
        else:
            print(f"{label} ({item[0]}, q[0]..q[n-1]): {item[1]}/{len(samples)}")


def compose_qiskit_circuits(qasm_paths):
    """Compose the three split QASM circuits into one Qiskit circuit."""
    circuits = [QuantumCircuit.from_qasm_file(str(path)).remove_final_measurements(inplace=False) for path in qasm_paths]
    num_qubits = circuits[0].num_qubits

    if any(circuit.num_qubits != num_qubits for circuit in circuits):
        raise ValueError("All split circuits must have the same number of qubits.")

    combined = QuantumCircuit(num_qubits)
    for circuit in circuits:
        combined = combined.compose(circuit)
    return combined


def run_aer_counts(qasm_paths, shots, method):
    """Run Aer on the composed Qiskit circuit for comparison."""
    from qiskit_aer import AerSimulator

    circuit = compose_qiskit_circuits(qasm_paths)
    circuit.measure_all()

    simulator = AerSimulator(method=method)
    transpiled = transpile(circuit, simulator)
    result = simulator.run(transpiled, shots=shots).result()
    return result.get_counts()


def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="autoray")
    args = parse_args()
    qasm_paths = [args.part1, args.part2, args.part3]
    state_max_bond = args.state_max_bond if args.state_max_bond is not None else args.max_bond
    state_cutoff = args.state_cutoff if args.state_cutoff is not None else args.cutoff

    args_dict = vars(args).copy()
    print("Running three MPO jobs in parallel:")
    for index, path in enumerate(qasm_paths, start=1):
        print(f"  part {index}: {path}")

    t0 = time.perf_counter()
    results = {}
    executor_cls = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
    try:
        executor = executor_cls(max_workers=3)
    except PermissionError as exc:
        if args.executor != "process":
            raise
        print(
            "ProcessPoolExecutor is not available in this environment "
            f"({exc}). Falling back to ThreadPoolExecutor for this run."
        )
        executor = ThreadPoolExecutor(max_workers=3)

    with executor:
        futures = {
            executor.submit(compress_one_part, index, path, args_dict): index
            for index, path in enumerate(qasm_paths, start=1)
        }
        for future in as_completed(futures):
            result = future.result()
            results[result["index"]] = result
            print(
                f"Finished part {result['index']}: "
                f"{result['num_qubits']} qubits, {result['num_ops']} ops, "
                f"{result['elapsed']:.3f}s, MPO info={result['mpo_info']}"
            )

    if len(results) != len(qasm_paths):
        raise RuntimeError(
            f"Expected {len(qasm_paths)} completed MPO parts, got {sorted(results)}."
        )
    ordered_results = [results[index] for index in sorted(results)]
    segment_mpos = [result["mpo"] for result in ordered_results]

    print("Applying segment MPOs to |0...0> in circuit order...")
    final_mps = apply_segment_mpos_to_zero_state(
        segment_mpos,
        mpo_max_bond=args.max_bond,
        mpo_cutoff=args.cutoff,
        state_max_bond=state_max_bond,
        state_cutoff=state_cutoff,
    )
    print("Final MPS info:", get_tn_info(final_mps))

    bitstring, p0s = extract_peak_bitstring(final_mps)
    print("Marginal-predicted bitstring (q[0]..q[n-1]):", bitstring)
    print("Marginal-predicted bitstring (Qiskit display order):", bitstring[::-1])
    print("Per-qubit P(0):", [round(p, 6) for p in p0s])

    if args.mps_shots > 0:
        samples = sample_tns(final_mps, args.mps_shots)
        print_sample_summary(samples, bitstring)

    if args.aer_shots > 0:
        print(f"Running Aer comparison with {args.aer_shots} shots...")
        counts = run_aer_counts(qasm_paths, args.aer_shots, args.aer_method)
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        print("Aer most frequent strings (Qiskit display order):")
        for bitstring_qiskit_order, count in ranked[:5]:
            print(f"  {bitstring_qiskit_order}: {count}/{args.aer_shots}")

    print(f"Total elapsed seconds: {time.perf_counter() - t0:.3f}")


if __name__ == "__main__":
    main()p