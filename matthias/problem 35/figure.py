from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def make_window_plots(
    csv_path="identity_window_slurm/identity_window_trace_overlap_problem35.csv",
    output_dir="figures",
    score_col=None,
):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)

    # Try to infer the score column if not provided
    if score_col is None:
        preferred = [
            c for c in df.columns
            if any(key in c.lower() for key in ["trace_overlap", "trace", "overlap", "score", "norm"])
        ]
        if preferred:
            score_col = preferred[0]
        else:
            raise ValueError(
                f"Could not infer score column automatically. Columns are: {list(df.columns)}"
            )

    if "window_size" not in df.columns or "cut_index" not in df.columns:
        raise ValueError(
            f"Expected columns 'window_size' and 'cut_index'. Columns are: {list(df.columns)}"
        )

    print(f"Using score column: {score_col}")

    for window_size, subdf in df.groupby("window_size"):
        subdf = subdf.sort_values("cut_index")

        plt.figure(figsize=(8, 4.5))
        plt.plot(subdf["cut_index"], subdf[score_col])
        plt.xlabel("cut index")
        plt.ylabel(score_col)
        plt.title(f"Window size = {window_size}")
        plt.tight_layout()

        outpath = output_dir / f"window_{int(window_size):03d}.png"
        plt.savefig(outpath, dpi=200)
        plt.close()

        print(f"Saved {outpath}")


if __name__ == "__main__":
    make_window_plots()

    import pandas as pd
    df = pd.read_csv("identity_window_slurm/identity_window_trace_overlap_problem35.csv")
    score_col = "identity_overlap"
    sub = df[df["window_size"] == 100]
    best = sub.loc[sub[score_col].idxmax()]
    print("best cut index:", int(best["cut_index"]))
    print("peak value:", best[score_col])
    print(best)