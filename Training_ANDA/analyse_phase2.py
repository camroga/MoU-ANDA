from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT = Path("../runs/mou-independent-anda-grid")

METRIC_PATH = (
    "00000-100-shot-obama-mirror-low_shot-kimg400-"
    "color-translation-cutout/"
    "metric-fid50k_full.jsonl"
)

OUTPUT = Path("../runs/analyse_phase2")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Phase 2 calibration grid
targets = [0.70]
k_values = [400, 500, 750]
epsilons = [0.15, 0.20, 0.25, 0.30]

EXPECTED_RUNS = (
    len(targets)
    * len(k_values)
    * len(epsilons)
)


# ---------------------------------------------------------
# READ EXPERIMENTS
# ---------------------------------------------------------

results = []

pattern = re.compile(
    r"target-(\d+p\d+)-"
    r"akimg-(\d+)-"
    r"epsilon-(\d+p\d+)-"
    r"seed-(\d+)"
)

for folder in ROOT.glob("target-*"):

    match = pattern.fullmatch(folder.name)

    if not match:
        continue

    target = float(
        match.group(1).replace("p", ".")
    )

    k_anda = int(match.group(2))

    epsilon = float(
        match.group(3).replace("p", ".")
    )

    seed = int(match.group(4))

    # Ignore experiments outside the current grid.
    if (
        target not in targets
        or k_anda not in k_values
        or epsilon not in epsilons
    ):
        continue

    metric_file = folder / METRIC_PATH

    if not metric_file.exists():
        print(
            "Missing metric file:",
            folder.name
        )
        continue

    fid_values = []

    with open(metric_file, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            data = json.loads(line)

            fid = data["results"]["fid50k_full"]
            snapshot = data["snapshot_pkl"]

            kimg = int(
                snapshot
                .replace(
                    "network-snapshot-",
                    ""
                )
                .replace(
                    ".pkl",
                    ""
                )
            )

            # Ignore initial network at 0 kimg.
            if kimg > 0:
                fid_values.append(
                    (kimg, fid)
                )

    if not fid_values:
        print(
            "No valid FID values:",
            folder.name
        )
        continue

    # Best FID observed during training.
    best_kimg, best_fid = min(
        fid_values,
        key=lambda x: x[1]
    )

    # FID exactly at 400 kimg.
    final = [
        fid
        for kimg, fid in fid_values
        if kimg == 400
    ]

    if not final:
        print(
            "Incomplete run:",
            folder.name
        )
        continue

    final_fid = final[0]

    results.append({
        "target": target,
        "K": k_anda,
        "epsilon": epsilon,
        "seed": seed,
        "min_fid": best_fid,
        "min_kimg": best_kimg,
        "final_fid": final_fid,
    })


# ---------------------------------------------------------
# VALIDATE RESULTS
# ---------------------------------------------------------

if not results:
    raise RuntimeError(
        "No completed Phase 2 experiments were found."
    )

if len(results) != EXPECTED_RUNS:
    print()
    print(
        "WARNING:"
        f" found {len(results)} completed runs,"
        f" expected {EXPECTED_RUNS}."
    )
    print()


# ---------------------------------------------------------
# CREATE TABLE
# ---------------------------------------------------------

df = pd.DataFrame(results)

df = df.sort_values(
    [
        "min_fid",
        "final_fid",
    ]
).reset_index(drop=True)

df.insert(
    0,
    "rank",
    range(1, len(df) + 1)
)

print("\nBest configurations:\n")

columns = [
    "rank",
    "target",
    "K",
    "epsilon",
    "min_fid",
    "min_kimg",
    "final_fid",
]

print(
    df[columns]
    .head(10)
    .to_string(index=False)
)

table_df = df[columns]


# ---------------------------------------------------------
# SAVE TABLES
# ---------------------------------------------------------

table_df.to_latex(
    OUTPUT / "phase2_results_table.tex",
    index=False,
    float_format="%.4f",
)

table_df.head(3).to_latex(
    OUTPUT / "phase2_top3_table.tex",
    index=False,
    float_format="%.4f",
)


# ---------------------------------------------------------
# HEATMAP FUNCTION
# ---------------------------------------------------------

def make_heatmap(
    column,
    title,
    filename
):

    matrix = (
        df
        .pivot(
            index="K",
            columns="epsilon",
            values=column
        )
        .reindex(
            index=k_values,
            columns=epsilons
        )
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        matrix.values,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(epsilons))
    )

    ax.set_xticklabels(
        epsilons
    )

    ax.set_yticks(
        range(len(k_values))
    )

    ax.set_yticklabels(
        k_values
    )

    ax.set_xlabel(
        r"$\epsilon$"
    )

    ax.set_ylabel(
        r"$K_{\mathrm{ANDA}}$"
    )

    # Write FID inside cells.
    for i in range(len(k_values)):

        for j in range(len(epsilons)):

            value = matrix.iloc[i, j]

            if not np.isnan(value):

                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center"
                )

    fig.colorbar(
        image,
        ax=ax,
        label="FID50k"
    )

    ax.set_title(title)

    fig.tight_layout()

    fig.savefig(
        OUTPUT / filename,
        bbox_inches="tight"
    )

    plt.close(fig)


# ---------------------------------------------------------
# CREATE FIGURES
# ---------------------------------------------------------

make_heatmap(
    "min_fid",
    "Best FID50k",
    "phase2_minimum_fid_heatmap.pdf"
)

make_heatmap(
    "final_fid",
    "FID50k at 400 kimg",
    "phase2_final_fid_heatmap.pdf"
)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print()
print(
    f"Completed experiments: "
    f"{len(results)}/{EXPECTED_RUNS}"
)

print(
    "Files created in:",
    OUTPUT
)

print()
print("Top 3 configurations:")
print(
    table_df
    .head(3)
    .to_string(index=False)
)