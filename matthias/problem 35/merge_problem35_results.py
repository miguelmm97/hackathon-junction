from csv import DictReader, DictWriter
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent / "identity_window_slurm"
PARTS_DIR = OUTPUT_DIR / "parts"
MERGED_PATH = OUTPUT_DIR / "identity_window_trace_overlap_problem35.csv"


def main():
    records = []
    for path in sorted(PARTS_DIR.glob("*.csv")):
        with open(path, newline="") as file:
            records.extend(DictReader(file))

    if not records:
        raise ValueError(f"No part files found in {PARTS_DIR}")

    records.sort(key=lambda row: (int(row["window_size"]), int(row["cut_index"])))

    with open(MERGED_PATH, "w", newline="") as file:
        writer = DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(MERGED_PATH)
    print(len(records))


if __name__ == "__main__":
    main()
