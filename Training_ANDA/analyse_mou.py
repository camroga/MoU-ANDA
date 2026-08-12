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
    "00000-obama-mirror-low_shot-kimg400-color-translation-cutout/"
    "metric-fid50k_full.jsonl"
)

OUTPUT = ROOT / "analysis_mou"
OUTPUT.mkdir(exist_ok=True)

# Current calibration grid
targets = [0.50, 0.55, 0.60, 0.65]
k_values = [500, 750, 1000]
epsilons = [0.10, 0.20, 0.30]

# ---------------------------------------------------------
# READ EXPERIMENTS
# ---------------------------------------------------------

results = []

pattern = re.compile(
    r"target-(\d+p\d+)-akimg-(\d+)-epsilon-(\d+p\d+)-seed-(\d+)"
)

for folder in ROOT.glob("target-*"):

    match = pattern.fullmatch(folder.name)

    if not match:
        continue

    target = float(match.group(1).replace("p", "."))
    k_anda = int(match.group(2))
    epsilon = float(match.group(3).replace("p", "."))
    seed = int(match.group(4))

    # Ignore experiments outside the current grid
    if (
        target not in targets
        or k_anda not in k_values
        or epsilon not in epsilons
    ):
        continue

    metric_file = folder / METRIC_PATH

    if not metric_file.exists():
        print("Missing:", folder.name)
        continue

    fid_values = []

    with open(metric_file, "r") as f:
        for line in f:

            data = json.loads(line)

            fid = data["results"]["fid50k_full"]
            snapshot = data["snapshot_pkl"]

            kimg = int(
                snapshot
                .replace("network-snapshot-", "")
                .replace(".pkl", "")
            )

            # Ignore initial network
            if kimg > 0:
                fid_values.append((kimg, fid))

    if not fid_values:
        continue

    # Minimum FID
    best_kimg, best_fid = min(
        fid_values,
        key=lambda x: x[1]
    )

    # FID exactly at 400 kimg
    final = [
        fid
        for kimg, fid in fid_values
        if kimg == 400
    ]

    if not final:
        print("Still running:", folder.name)
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
# CREATE TABLE
# ---------------------------------------------------------

df = pd.DataFrame(results)

df = df.sort_values(
    ["min_fid", "final_fid"]
).reset_index(drop=True)

df.insert(0, "rank", range(1, len(df) + 1))

print("\nBest configurations:\n")

print(
    df[
        [
            "rank",
            "target",
            "K",
            "epsilon",
            "min_fid",
            "min_kimg",
            "final_fid",
        ]
    ].head(10).to_string(index=False)
)

table_df = df[
    [
        "rank",
        "target",
        "K",
        "epsilon",
        "min_fid",
        "min_kimg",
        "final_fid",
    ]
]

# Complete ranked table
table_df.to_latex(
    OUTPUT / "mou_results_table.tex",
    index=False,
    float_format="%.4f",
)

# Top 3 only
table_df.head(3).to_latex(
    OUTPUT / "mou_top3_table.tex",
    index=False,
    float_format="%.4f",
)

# ---------------------------------------------------------
# HEATMAP FUNCTION
# ---------------------------------------------------------

def make_heatmap(column, title, filename):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4),
        sharey=True
    )

    # Same FID scale in the three panels
    vmin = df[column].min()
    vmax = df[column].max()

    image = None

    for ax, epsilon in zip(axes, epsilons):

        subset = df[df["epsilon"] == epsilon]

        matrix = (
            subset
            .pivot(
                index="K",
                columns="target",
                values=column
            )
            .reindex(
                index=k_values,
                columns=targets
            )
        )

        image = ax.imshow(
            matrix.values,
            vmin=vmin,
            vmax=vmax,
            aspect="auto"
        )

        ax.set_title(
            rf"$\epsilon={epsilon:.2f}$"
        )

        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets)

        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels(k_values)

        ax.set_xlabel(
            r"$t_{\mathrm{ANDA}}$"
        )

        # Write FID inside cells
        for i in range(len(k_values)):
            for j in range(len(targets)):

                value = matrix.iloc[i, j]

                if not np.isnan(value):
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center"
                    )

    axes[0].set_ylabel(
        r"$K_{\mathrm{ANDA}}$"
    )

    fig.colorbar(
        image,
        ax=axes,
        label="FID"
    )

    fig.suptitle(title)

    plt.savefig(
        OUTPUT / filename,
        bbox_inches="tight"
    )

    plt.close()

# ---------------------------------------------------------
# CREATE THE TWO FIGURES
# ---------------------------------------------------------

make_heatmap(
    "min_fid",
    "Minimum FID",
    "minimum_fid_heatmap.pdf"
)

make_heatmap(
    "final_fid",
    "FID at 400 kimg",
    "final_fid_heatmap.pdf"
)

print("\nFiles created in:", OUTPUT)