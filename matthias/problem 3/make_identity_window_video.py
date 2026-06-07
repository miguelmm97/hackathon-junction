import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from qiskit import QuantumCircuit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matthias.cut_scan.cut_scan import (
    identity_score,
    identity_window_mpo,
    problem_qasm_path,
    qasm_gate_line_numbers,
    selected_cut_indices,
)


PROBLEM_NUMBER = 3
WINDOW_SIZES = list(range(1, 101))
STRIDE = 1
MAX_BOND = 32
CUTOFF = 1e-4
FPS = 30
N_JOBS = int(os.environ.get("IDENTITY_VIDEO_N_JOBS", "4"))

OUTPUT_DIR = Path(__file__).resolve().parent / f"identity_window_video_B{MAX_BOND}"
PROFILE_DIR = OUTPUT_DIR / "profiles"
FRAME_DIR = OUTPUT_DIR / "frames"
VIDEO_PATH = OUTPUT_DIR / "identity_window_trace_W001_W100.mp4"


def profile_path(window_size):
    return PROFILE_DIR / f"profile_W{window_size:03d}.csv"


def frame_path(window_size):
    return FRAME_DIR / f"frame_W{window_size:03d}.png"


def write_profile(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def read_profile(path):
    with open(path, newline="") as file:
        rows = []
        for row in csv.DictReader(file):
            rows.append(
                {
                    "cut_index": int(row["cut_index"]),
                    "qasm_line": int(row["qasm_line"]),
                    "identity_overlap": float(row["identity_overlap"]),
                }
            )
    return rows


def compute_profile(window_size):
    path = profile_path(window_size)
    if path.exists():
        return str(path)

    qasm_path = problem_qasm_path(PROBLEM_NUMBER)
    circuit = QuantumCircuit.from_qasm_file(qasm_path)
    gate_lines = qasm_gate_line_numbers(qasm_path)

    if len(gate_lines) != len(circuit):
        raise ValueError("QASM gate lines do not match circuit instructions")

    cut_indices = selected_cut_indices(len(circuit), window_size, STRIDE)
    records = []

    for cut_index in cut_indices:
        started = time.perf_counter()
        mpo = identity_window_mpo(
            circuit,
            cut_index,
            window_size,
            MAX_BOND,
            CUTOFF,
            run_swap_optimization=False,
        )
        records.append(
            {
                "window_size": window_size,
                "cut_index": cut_index,
                "qasm_line": gate_lines[cut_index],
                "identity_overlap": identity_score(mpo, circuit.num_qubits, "trace_overlap"),
                "runtime_seconds": time.perf_counter() - started,
            }
        )

    write_profile(path, records)
    return str(path)


def render_frame(window_size):
    records = read_profile(profile_path(window_size))
    path = frame_path(window_size)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    x = [row["cut_index"] - window_size for row in records]
    y = [row["identity_overlap"] for row in records]

    plt.rcParams.update({"text.usetex": True})
    fig, ax = plt.subplots(figsize=(4, 4), dpi=240)
    ax.plot(x, y, color="#005f63", linewidth=1.8)
    ax.set_xlabel(r"Layer index $i$")
    ax.set_ylabel(r"$\mathrm{Tr}(\mathrm{MPO}_W)$")
    ax.set_xlim(min(x), max(x))
    ax.margins(x=0)
    ax.text(0.03, 0.93, r"$W=%d$" % window_size, transform=ax.transAxes, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def make_video():
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        str(FRAME_DIR / "frame_W%03d.png"),
        "-vf",
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "slow",
        str(VIDEO_PATH),
    ]
    subprocess.run(command, check=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Computing profiles with n_jobs={N_JOBS}, max_bond={MAX_BOND}")
    Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(compute_profile)(window_size) for window_size in WINDOW_SIZES
    )

    print("Rendering frames")
    for window_size in WINDOW_SIZES:
        render_frame(window_size)

    print("Writing mp4")
    make_video()
    print(VIDEO_PATH)


if __name__ == "__main__":
    main()
