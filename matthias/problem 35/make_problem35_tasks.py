from csv import DictWriter
from math import ceil
from pathlib import Path


PROBLEM_NUMBER = 35
WINDOW_SIZES = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    80,
    100,
]
STRIDE = 1
MAX_BOND = 128
CUTOFF = 1e-4
CUTS_PER_ARRAY_TASK = 128

OUTPUT_DIR = Path(__file__).resolve().parent / "identity_window_slurm"
TASKS_PATH = OUTPUT_DIR / "tasks.csv"


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
    return list(range(first, last + 1, stride))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    qasm_files = sorted(Path(__file__).resolve().parent.glob("*.qasm"))
    if len(qasm_files) != 1:
        raise ValueError(f"Expected one QASM file, found {len(qasm_files)}")
    num_gates = len(qasm_gate_line_numbers(qasm_files[0]))

    for window_size in WINDOW_SIZES:
        cut_indices = selected_cut_indices(num_gates, window_size, STRIDE)
        num_chunks = ceil(len(cut_indices) / CUTS_PER_ARRAY_TASK)

        for chunk_id in range(num_chunks):
            start = chunk_id * CUTS_PER_ARRAY_TASK
            stop = min(start + CUTS_PER_ARRAY_TASK, len(cut_indices))
            rows.append(
                {
                    "task_id": len(rows),
                    "problem_number": PROBLEM_NUMBER,
                    "window_size": window_size,
                    "chunk_id": chunk_id,
                    "num_chunks": num_chunks,
                    "start": start,
                    "stop": stop,
                    "cut_count": stop - start,
                    "stride": STRIDE,
                    "max_bond": MAX_BOND,
                    "cutoff": CUTOFF,
                }
            )

    with open(TASKS_PATH, "w", newline="") as file:
        writer = DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(TASKS_PATH)
    print(len(rows))


if __name__ == "__main__":
    main()
