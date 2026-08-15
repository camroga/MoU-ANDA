#!/usr/bin/env python3

import json
import glob
import os
import matplotlib.pyplot as plt

BASE_DIR = "/workspace/thesis/calibration/runs/mou-target-recalibration"
OUTPUT_DIR = "/workspace/thesis/calibration/runs/analyse_phase1"

TARGETS = {
    "0p70": "0.70",
    "0p80": "0.80",
    "0p90": "0.90",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.figure(figsize=(8, 5))

found = 0

for target_dir, target_label in TARGETS.items():

    pattern = os.path.join(
        BASE_DIR,
        f"target-{target_dir}-akimg-400-epsilon-0p20-seed-1",
        "00000-*",
        "stats.jsonl",
    )

    files = glob.glob(pattern)

    if not files:
        print(f"[WARNING] No stats.jsonl found for target {target_label}")
        continue

    stats_file = files[0]

    kimgs = []
    uncertainty_pct = []

    with open(stats_file, "r") as f:
        for line in f:
            data = json.loads(line)

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
        f"{len(kimgs)} points | "
        f"last kimg={kimgs[-1]:.3f} | "
        f"uncertainty={uncertainty_pct[-1]:.3f}%"
    )

if found == 0:
    raise RuntimeError("No Phase 1 stats.jsonl files found.")

plt.xlabel("Training progress (kimg)")
plt.ylabel("Observations within MoU (%)")
plt.title("Margin of Uncertainty During ANDA Target Recalibration")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

pdf_path = os.path.join(
    OUTPUT_DIR,
    "phase1_mou_uncertainty.pdf"
)

plt.savefig(
    pdf_path,
    bbox_inches="tight"
)

print("\nAnalysis completed.")
print(f"PDF: {pdf_path}")