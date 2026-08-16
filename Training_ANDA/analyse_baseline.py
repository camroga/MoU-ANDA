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


def get_final_fid(dataset, seed):

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

    results = []

    for metric_file in files:
        with open(metric_file, "r") as f:
            for line in f:
                data = json.loads(line)

                if data.get("metric") == "fid50k_full":
                    results.append(
                        data["results"]["fid50k_full"]
                    )

    if not results:
        raise RuntimeError(
            f"No FID results found for {dataset}, seed {seed}"
        )

    return results[-1]


# --------------------------------------------------
# Collect results
# --------------------------------------------------

results = {}

for dataset in DATASETS:

    values = []

    for seed in SEEDS:
        fid = get_final_fid(dataset, seed)
        values.append(fid)

    results[dataset] = {
        "seeds": values,
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
    }


# --------------------------------------------------
# Console table
# --------------------------------------------------

print("\nBaseline FID50k results\n")

print(
    f"{'Dataset':<10}"
    f"{'Seed 2':>12}"
    f"{'Seed 3':>12}"
    f"{'Seed 4':>12}"
    f"{'Mean':>12}"
    f"{'Std':>12}"
)

print("-" * 70)

for dataset in DATASETS:

    r = results[dataset]

    print(
        f"{dataset.capitalize():<10}"
        f"{r['seeds'][0]:>12.4f}"
        f"{r['seeds'][1]:>12.4f}"
        f"{r['seeds'][2]:>12.4f}"
        f"{r['mean']:>12.4f}"
        f"{r['std']:>12.4f}"
    )


# --------------------------------------------------
# Detailed LaTeX table
# --------------------------------------------------

latex_detail = os.path.join(
    OUTPUT_DIR,
    "baseline_fid_results.tex"
)

with open(latex_detail, "w") as f:

    f.write("\\begin{table}[t]\n")
    f.write("\\centering\n")
    f.write("\\caption{Baseline FID50k results across three random seeds.}\n")
    f.write("\\label{tab:baseline_fid}\n")
    f.write("\\begin{tabular}{lccccc}\n")
    f.write("\\hline\n")
    f.write("Dataset & Seed 2 & Seed 3 & Seed 4 & Mean & Std. \\\\\n")
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


# --------------------------------------------------
# Final comparison table
# --------------------------------------------------

comparison_path = os.path.join(
    OUTPUT_DIR,
    "method_comparison.tex"
)

obama = results["obama"]
panda = results["panda"]

with open(comparison_path, "w") as f:

    f.write("\\begin{table}[t]\n")
    f.write("\\centering\n")
    f.write(
        "\\caption{FID comparison between the baseline and the proposed method. Lower is better.}\n"
    )
    f.write("\\label{tab:method_comparison}\n")
    f.write("\\begin{tabular}{lcc}\n")
    f.write("\\hline\n")
    f.write("Method & Obama & Panda \\\\\n")
    f.write("\\hline\n")

    f.write(
        f"StyleGAN2 + ADA + ANDA & "
        f"{obama['mean']:.2f} $\\pm$ {obama['std']:.2f} & "
        f"{panda['mean']:.2f} $\\pm$ {panda['std']:.2f} \\\\\n"
    )

    f.write(
        "Proposed MoU-ANDA & -- & -- \\\\\n"
    )

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")


print("\nAnalysis completed.")
print(f"LaTeX: {latex_detail}")
print(f"LaTeX: {comparison_path}")