#!/usr/bin/env python3

import os
import glob
import json
import statistics

BASE_DIR = "/workspace/thesis/baseline/runs/anda-original"
OUTPUT_DIR = "/workspace/thesis/baseline/runs/analyse_baseline"

DATASETS = ["obama", "panda"]
SEEDS = [2, 3, 4]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_fids(dataset, seed):

    pattern = os.path.join(
        BASE_DIR,
        f"anda-original-{dataset}-seed-{seed}",
        "**",
        "metric-fid50k_full.jsonl",
    )

    files = glob.glob(pattern, recursive=True)

    if not files:
        raise FileNotFoundError(
            f"No metric file found for {dataset}, seed {seed}"
        )

    records = []

    for metric_file in files:
        with open(metric_file, "r") as f:
            for line in f:
                data = json.loads(line)

                if data.get("metric") == "fid50k_full":
                    records.append({
                        "fid": data["results"]["fid50k_full"],
                        "snapshot": data.get("snapshot_pkl", ""),
                        "timestamp": data.get("timestamp", 0),
                    })

    if not records:
        raise RuntimeError(
            f"No FID results found for {dataset}, seed {seed}"
        )

    # Ensure chronological order.
    records.sort(key=lambda x: x["timestamp"])

    best = min(records, key=lambda x: x["fid"])
    final = records[-1]

    return best, final


# --------------------------------------------------
# Collect results
# --------------------------------------------------

best_results = {}
final_results = {}

for dataset in DATASETS:

    best_values = []
    final_values = []

    for seed in SEEDS:

        best, final = get_fids(dataset, seed)

        best_values.append(best["fid"])
        final_values.append(final["fid"])

        print(
            f"{dataset.capitalize()} seed {seed} | "
            f"best={best['fid']:.4f} ({best['snapshot']}) | "
            f"final={final['fid']:.4f} ({final['snapshot']})"
        )

    best_results[dataset] = {
        "seeds": best_values,
        "mean": statistics.mean(best_values),
        "std": statistics.stdev(best_values),
    }

    final_results[dataset] = {
        "seeds": final_values,
        "mean": statistics.mean(final_values),
        "std": statistics.stdev(final_values),
    }


def write_table(results, filename, caption, label):

    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w") as f:

        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\hline\n")
        f.write(
            "Dataset & Seed 2 & Seed 3 & Seed 4 & Mean & Std. \\\\\n"
        )
        f.write("\\hline\n")

        for dataset in DATASETS:

            r = results[dataset]

            f.write(
                f"{dataset.capitalize()} & "
                f"{r['seeds'][0]:.2f} & "
                f"{r['seeds'][1]:.2f} & "
                f"{r['seeds'][2]:.2f} & "
                f"{r['mean']:.2f} & "
                f"{r['std']:.2f} \\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    return path


best_path = write_table(
    best_results,
    "baseline_best_fid.tex",
    "Best FID50k obtained during training across three random seeds.",
    "tab:baseline_best_fid",
)

final_path = write_table(
    final_results,
    "baseline_final_fid.tex",
    "Final FID50k at 500 kimg across three random seeds.",
    "tab:baseline_final_fid",
)

print("\nAnalysis completed.")
print(f"Best FID table:  {best_path}")
print(f"Final FID table: {final_path}")