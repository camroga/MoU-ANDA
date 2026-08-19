#!/usr/bin/env python3

import json
import glob
import os
import matplotlib.pyplot as plt

BASE_DIR = "/workspace/thesis/calibration/runs/mou-target-recalibration-inverted"
OUTPUT_DIR = "/workspace/thesis/calibration/runs/analyse_phase1_inverted"

TARGETS = {
    "0p50": "0.50",
    "0p60": "0.60",
    "0p70": "0.70",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.figure(figsize=(8, 5))

found = 0
results = []

for target_dir, target_label in TARGETS.items():

    pattern = os.path.join(
        BASE_DIR,
        f"target-{target_dir}-akimg-400-epsilon-0p20-seed-1",
        "00000-*",
    )

    experiment_dirs = glob.glob(pattern)

    if not experiment_dirs:
        print(f"[WARNING] No experiment found for target {target_label}")
        continue

    experiment_dir = experiment_dirs[0]

    stats_file = os.path.join(
        experiment_dir,
        "stats.jsonl"
    )

    metric_file = os.path.join(
        experiment_dir,
        "metric-fid50k_full.jsonl"
    )

    if not os.path.exists(stats_file):
        print(f"[WARNING] No stats.jsonl found for target {target_label}")
        continue

    if not os.path.exists(metric_file):
        print(f"[WARNING] No FID file found for target {target_label}")
        continue

    # --------------------------------------------------
    # Read training statistics
    # --------------------------------------------------

    stats = []
    kimgs = []
    uncertainty_pct = []

    with open(stats_file, "r") as f:
        for line in f:
            data = json.loads(line)
            stats.append(data)

            if (
                "Progress/kimg" not in data
                or "MoU/uncertain" not in data
            ):
                continue

            kimgs.append(
                data["Progress/kimg"]["mean"]
            )

            uncertainty_pct.append(
                data["MoU/uncertain"]["mean"] * 100
            )

    if not kimgs:
        print(f"[WARNING] No MoU data found for target {target_label}")
        continue

    last_stats = stats[-1]

    final_p_anda = (
        last_stats["Progress/anda_p"]["mean"] * 100
    )

    nda_applied = (
        last_stats["ANDA/applied"]["mean"] * 100
    )

    mou_uncertain = (
        last_stats["MoU/uncertain"]["mean"] * 100
    )

    # --------------------------------------------------
    # Read FID results
    # --------------------------------------------------

    fid_records = []

    with open(metric_file, "r") as f:
        for line in f:
            data = json.loads(line)

            if data.get("metric") == "fid50k_full":
                fid_records.append(data)

    if not fid_records:
        print(f"[WARNING] No FID values found for target {target_label}")
        continue

    fid_records.sort(
        key=lambda x: x.get("timestamp", 0)
    )

    best_record = min(
        fid_records,
        key=lambda x: x["results"]["fid50k_full"]
    )

    final_record = fid_records[-1]

    best_fid = (
        best_record["results"]["fid50k_full"]
    )

    final_fid = (
        final_record["results"]["fid50k_full"]
    )

    # --------------------------------------------------
    # Save summary
    # --------------------------------------------------

    results.append({
        "target": target_label,
        "best_fid": best_fid,
        "final_fid": final_fid,
        "final_p_anda": final_p_anda,
        "nda_applied": nda_applied,
        "mou_uncertain": mou_uncertain,
    })

    # --------------------------------------------------
    # Plot MoU uncertainty
    # --------------------------------------------------

    plt.plot(
        kimgs,
        uncertainty_pct,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label=f"Target = {target_label}",
    )

    found += 1

    print(
        f"Target {target_label}: "
        f"Best FID={best_fid:.4f} | "
        f"Final FID={final_fid:.4f} | "
        f"Final p_anda={final_p_anda:.2f}% | "
        f"NDA applied={nda_applied:.2f}% | "
        f"MoU uncertain={mou_uncertain:.2f}%"
    )


if found == 0:
    raise RuntimeError("No inverted Phase 1 experiment files found.")


# ======================================================
# MoU uncertainty figure
# ======================================================

plt.xlabel("Training progress (kimg)")
plt.ylabel("Observations within MoU (%)")
plt.title("Margin of Uncertainty During ANDA Target Recalibration")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

pdf_path = os.path.join(
    OUTPUT_DIR,
    "phase1_inverted_mou_uncertainty.pdf"
)

plt.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.close()


# ======================================================
# Function to generate LaTeX tables
# ======================================================

def write_table(data, filename, caption, label):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    with open(path, "w") as f:

        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{tabular}{cccccc}\n")
        f.write("\\hline\n")

        f.write(
            "Target & Best FID & Final FID & "
            "Final $p_{ANDA}$ (\\%) & "
            "NDA applied (\\%) & "
            "MoU uncertain (\\%) \\\\\n"
        )

        f.write("\\hline\n")

        for r in data:

            f.write(
                f"{r['target']} & "
                f"{r['best_fid']:.2f} & "
                f"{r['final_fid']:.2f} & "
                f"{r['final_p_anda']:.2f} & "
                f"{r['nda_applied']:.2f} & "
                f"{r['mou_uncertain']:.2f} \\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    return path


# ======================================================
# Best FID table
# ======================================================

best_results = sorted(
    results,
    key=lambda x: x["best_fid"]
)

best_table = write_table(
    best_results,
    "phase1_inverted_best_fid.tex",
    "Phase 1 target recalibration results ordered by best FID50k.",
    "tab:phase1_inverted_best_fid",
)


# ======================================================
# Final FID table
# ======================================================

final_results = sorted(
    results,
    key=lambda x: x["final_fid"]
)

final_table = write_table(
    final_results,
    "phase1_inverted_final_fid.tex",
    "Phase 1 target recalibration results ordered by final FID50k.",
    "tab:phase1_inverted_final_fid",
)


# ======================================================
# Final output
# ======================================================

print("\nAnalysis completed.")
print(f"PDF:             {pdf_path}")
print(f"Best FID table:  {best_table}")
print(f"Final FID table: {final_table}")