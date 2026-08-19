# MoU-ANDA: Margin-of-Uncertainty-Based Adaptive Negative Data Augmentation

This repository contains the implementation and experimental evaluation of **MoU-ANDA**, an extension of Adaptive Negative Data Augmentation (ANDA) for data-efficient GAN training.

The proposed method introduces a **Margin of Uncertainty (MoU)** around the discriminator logits obtained from real images. While the original ANDA feedback mechanism directly uses the sign of these logits, MoU introduces an additional uncertain region around the decision boundary.

For real images:

- logits above the uncertainty margin indicate that the discriminator classifies the sample toward the real side;
- logits within the uncertainty margin are treated as uncertain;
- logits below the negative margin indicate that the sample is classified toward the fake side.

This produces a ternary feedback signal that is used by an independent controller to regulate the probability of applying negative-data augmentation during training.

This project was developed as part of a Master's thesis investigating whether a more detailed interpretation of discriminator behavior can be used as an alternative feedback mechanism for ANDA.

---

## Background

This repository is a **fork of the original ANDA project** and extends its implementation with the proposed MoU-ANDA mechanism.

The original ANDA method was introduced in:

> Z. Zhang, Y. Hua, G. Sun, H. Wang, and S. McLoone,  
> **"Improving the Leaking of Augmentations in Data-Efficient GANs via Adaptive Negative Data Augmentation,"**  
> Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024.

The original ANDA code was developed using implementations based on DiffAugment, StyleGAN2-ADA, and related data-efficient GAN approaches.

Related repositories:

- [DiffAugment](https://github.com/mit-han-lab/data-efficient-gans)
- [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch)
- [DeceiveD / APA](https://github.com/EndlessSora/DeceiveD)

This fork preserves the original ANDA code and functionality while adding the MoU-based feedback mechanism, the independent adaptive controller used in this thesis, the experimental execution scripts, and the analysis files associated with the calibration stages.

---

## Proposed MoU-ANDA Mechanism

The original ANDA feedback signal is derived from the sign of the discriminator logits for real images.

MoU-ANDA introduces an uncertainty margin controlled by `epsilon`. Instead of immediately reducing every discriminator response to a positive or negative value, logits close to the decision boundary are assigned an additional uncertain state.

The MoU feedback therefore distinguishes between:

1. real samples classified toward the real side;
2. real samples whose discriminator logits fall within the uncertainty margin;
3. real samples classified toward the fake side.

This ternary signal is then used by an independent controller to regulate the probability of applying ANDA.

The main MoU-ANDA hyperparameters evaluated in this work are:

- `t_ANDA` — target value used by the adaptive controller;
- `K_ANDA` — adaptation speed of the controller;
- `epsilon` — width of the Margin of Uncertainty;
- `lambda` — negative-sample mixture weight.

---

## Repository Structure

```text
MoU-ANDA/
├── Training_ANDA/
│   ├── training/
│   │   ├── loss.py
│   │   ├── training_loop.py
│   │   └── ...
│   │
│   └── execution/
│       ├── executer_target.sh
│       ├── executer_phase2.sh
│       ├── executer_phase3.sh
│       └── executer_final_validation.sh
│
├── analysis/
│   ├── phase1/
│   │   ├── analyse_phase1.py
│   │   ├── phase1_best_fid.tex
│   │   ├── phase1_final_fid.tex
│   │   └── phase1_mou_uncertainty.pdf
│   │
│   └── phase2/
│       ├── analyse_phase2.py
│       ├── phase2_final_fid_heatmap.pdf
│       ├── phase2_minimum_fid_heatmap.pdf
│       ├── phase2_results_table.tex
│       └── phase2_top3_table.tex
│
├── runs/
└── README.md
```

### `Training_ANDA/training`

Contains the main training implementation, including:

- the MoU feedback signal;
- the independent ANDA probability controller;
- the construction of negative-data-augmented samples;
- the GAN training loop.

### `Training_ANDA/execution`

Contains the scripts used for the main experimental stages:

- `executer_target.sh` — Phase 1: calibration of the ANDA target;
- `executer_phase2.sh` — Phase 2: calibration of `K_ANDA` and `epsilon`;
- `executer_phase3.sh` — Phase 3: calibration of the negative-sample mixture weight `lambda`;
- `executer_final_validation.sh` — final multi-seed evaluation on the Obama and Panda datasets.

> **Note:** Some execution scripts contain paths configured for the RunPod environment used during the experiments. Dataset and output paths may need to be modified before running them on another system.

### `analysis`

Contains the scripts and generated outputs used to analyze the calibration experiments.

The Phase 1 and Phase 2 directories contain the corresponding analysis scripts, LaTeX tables, and figures used during the experimental evaluation.

---

## Datasets

The experiments conducted for MoU-ANDA used two low-shot datasets:

- **100-shot Obama**
- **100-shot Panda**

Each dataset contains 100 training images and was used under the same low-shot training setting.

During the experiments, the datasets were stored in StyleGAN-compatible ZIP format as:

```text
100-shot-obama.zip
100-shot-panda.zip
```

The **100-shot Obama dataset** was used during the hyperparameter calibration stages.

Both **100-shot Obama** and **100-shot Panda** were used during the final multi-seed evaluation to examine the behavior of the selected MoU-ANDA configuration across datasets.

The low-shot dataset collection provided with the original ANDA work can be obtained from:

[Low-shot datasets](https://drive.google.com/file/d/1rWqaVlms55604jrP5t9ShacL6mZKWL8f/view?usp=sharing)

---

## Requirements

The training environment follows the dependencies required by the original ANDA and StyleGAN2-ADA implementations.

Please refer to:

- [DiffAugment](https://github.com/mit-han-lab/data-efficient-gans)
- [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch)

for the underlying environment requirements.

The MoU-ANDA experiments reported in this project were executed using NVIDIA RTX A6000 GPUs.

---

## Experimental Protocol

The MoU-ANDA hyperparameters were calibrated sequentially to reduce the experimental search space.

### Phase 1 — ANDA Target Calibration

The first stage evaluates the target value `t_ANDA` used by the MoU-based controller.

This stage also examines the behavior of the Margin of Uncertainty and verifies whether discriminator logits produced for real images fall within the defined uncertainty region during training.

### Phase 2 — MoU Hyperparameter Calibration

After selecting the target value, the second stage evaluates combinations of:

```text
K_ANDA
epsilon
```

where:

- `K_ANDA` controls the adaptation speed;
- `epsilon` defines the uncertainty margin around the discriminator logits.

Configurations are evaluated using both the best FID achieved during training and the FID obtained at the end of training.

### Phase 3 — Negative-Sample Mixture-Weight Calibration

The third stage evaluates the negative-sample mixture weight:

```text
lambda ∈ {0.15, 0.20, 0.25, 0.30}
```

The mixture weight determines the proportion of pseudo-data used when constructing an augmented negative sample.

### Final Validation

The MoU-ANDA configuration selected through the calibration procedure was:

```text
t_ANDA  = 0.70
K_ANDA  = 750
epsilon = 0.25
lambda  = 0.20
```

The final validation was performed using:

```text
Seeds:    2, 3, 4
Training: 500 kimg
Metric:   FID50k
```

on both:

```text
100-shot Obama
100-shot Panda
```

The same datasets, random seeds, training duration, and evaluation metric were used for the original ANDA baseline comparison.

---

## Evaluation

FID50k is used as the main quantitative evaluation metric.

Two FID measurements are considered throughout the calibration experiments:

- **Best FID** — the lowest FID observed during training;
- **Final FID** — the FID obtained at the end of the specified training period.

For the final multi-seed evaluation, results are reported using the mean and standard deviation across three independent random seeds.

To evaluate a trained network using the original evaluation utilities:

```bash
python calc_metrics.py \
    --metrics=fid50k_full \
    --data=<which-dataset> \
    --network=<which-pretrained>
```

---

## Image Generation

To generate images using a trained or pre-trained model:

```bash
python generate.py \
    --outdir=out \
    --seeds=1-16 \
    --network=<which-pretrained>
```

To generate GIFs:

```bash
python generate_gif.py \
    --output=<which-dataset>.gif \
    --seed=0 \
    --num-rows=1 \
    --num-cols=8 \
    --network=<which-pretrained>
```

---

## Original ANDA Pre-trained Models

The following links correspond to the **pre-trained models provided by the authors of the original ANDA project**. They are preserved here as references to the upstream repository and should not be interpreted as models trained as part of the MoU-ANDA thesis experiments.

### ANDA with StyleGAN2 + DiffAugment on Low-shot Datasets

- [100-shot Obama](https://drive.google.com/file/d/1gGNKasAsnDbBJ01h40s8x4KN-jmrKda7/view?usp=sharing)
- [100-shot Panda](https://drive.google.com/file/d/1t7LkDajXx_Mf49Sp4dRDlaZVCxXd7CSs/view?usp=sharing)
- [100-shot Grumpy Cat](https://drive.google.com/file/d/1wrWRgh-l-KsRtX8P22QVK3K2Ub8SY3nC/view?usp=sharing)
- [AnimalFace Cat](https://drive.google.com/file/d/1mb6wZaEg-rybVVG3PDyYe8sFAq17k6dE/view?usp=sharing)
- [AnimalFace Dog](https://drive.google.com/file/d/1RaDkC2Y0jwIAHbwSBDwgSG1N9VEvJoat/view?usp=sharing)

### ANDA with StyleGAN2 + ADA on Low-shot Datasets

- [100-shot Obama](https://drive.google.com/file/d/1WBVWypVyUp4Qg9WAhquo7Qgp3WAbwYuI/view?usp=sharing)
- [100-shot Panda](https://drive.google.com/file/d/1MaQjmb_mlsQfbuQtQxrLkVXmHgXwj-A_/view?usp=sharing)
- [100-shot Grumpy Cat](https://drive.google.com/file/d/1Ste68t4umvRtcR2lSrv_yqDrkkp85yus/view?usp=sharing)
- [AnimalFace Cat](https://drive.google.com/file/d/1zv6zmlcuc4G8SjT-iyn28AREy327WmxK/view?usp=sharing)
- [AnimalFace Dog](https://drive.google.com/file/d/1x5dS4mLy4dIga8GZNvYY938ClGR6rEQY/view?usp=sharing)

### ANDA with InsGen on Low-shot Datasets

- [100-shot Obama](https://drive.google.com/file/d/1MAKPfPNzPdrDhkCwqZEBJuZDAZq7ebIU/view?usp=sharing)
- [100-shot Panda](https://drive.google.com/file/d/1aM3G17Aqzvh2D-z45RvDt0znczjI_wXn/view?usp=sharing)
- [100-shot Grumpy Cat](https://drive.google.com/file/d/1hu-SUNIlKdrNSeJMueYyeeG75ysHJGUE/view?usp=sharing)
- [AnimalFace Cat](https://drive.google.com/file/d/1oHWmnwqB-ZF_Y0RfUJ3nyMe0DxfnZBbc/view?usp=sharing)
- [AnimalFace Dog](https://drive.google.com/file/d/1h4hFqavloaev34y2dwTgKa1n4A5ErdZm/view?usp=sharing)

---

## Notes on Original ANDA Evaluation

The original ANDA README reports that its pre-trained models were evaluated on an Alienware R8 desktop running Ubuntu 20.04 with an NVIDIA 2080 Ti GPU.

FID values may vary slightly depending on the GPU, software environment, and system used for evaluation.

This note refers to the evaluation environment of the **original ANDA project**, not to the MoU-ANDA experiments described above.

---

## Original ANDA Citation

If you use the original ANDA method, implementation, or pre-trained models, please cite:

```bibtex
@inproceedings{zhang2024improving,
  title={Improving the Leaking of Augmentations in Data-Efficient GANs via Adaptive Negative Data Augmentation},
  author={Zhang, Zhaoyu and Hua, Yang and Sun, Guanxiong and Wang, Hui and McLoone, Se{\'a}n},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5412--5421},
  year={2024}
}
```

---

## Acknowledgments

This repository was developed as a **fork of the original ANDA implementation**.

The MoU-based feedback mechanism, independent adaptive controller, experimental execution pipeline, and calibration analysis files were added as part of this Master's thesis.

This project also builds upon StyleGAN2-ADA, DiffAugment, and related data-efficient GAN implementations. The original repositories and publications should be consulted for their corresponding implementations, licenses, and research contributions.