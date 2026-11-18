# AgMA-Diff: Agent-Guided, Mode-Aware Diffusion Augmentation for Class-Imbalanced Diabetic Retinopathy Grading

This repository contains two notebooks for diabetic retinopathy (DR) grading on the **DDR** dataset:

1. **A reproduction of the MICCAI 2025 base paper**, *Class-Conditioned Image Synthesis with Diffusion for Imbalanced Diabetic Retinopathy Grading* (Zhang et al.).
2. **AgMA-Diff**, our proposed method. It extends the base paper with:
   - intra-grade **mode discovery**
   - a single **mode-conditioned Stable Diffusion LoRA**
   - a **three-gate verifier** for semantic consistency, realism and novelty
   - a **closed-loop, QWK-constrained augmentation agent**

Both notebooks use the **same DDR split, semantic scorer, LANet classifier, training schedule and synthetic-image budget**, so their results can be compared directly.

---

## Motivation

DDR is strongly imbalanced: the severe grade is **26.6× rarer** than the no-DR grade in the training set. The base paper shows that diffusion-generated minority images raise balanced accuracy and minority-grade recall, but they also lower quadratic weighted kappa (QWK) and accuracy. Our reproduction confirms this trade-off. AgMA-Diff aims to improve minority grades **without** reducing ordinal agreement. It does this by:

- deciding *which* and *how many* synthetic images to add
- verifying each image for class consistency, realism and novelty
- committing a batch only if a validation objective does not drop and QWK is preserved

---

## Repository structure

```
AgMA-Diff/
├── 01_base_paper_reproduction_local.ipynb   # reproduction of Zhang et al., MICCAI 2025
├── 02_AgMA_Diff_full_method_local.ipynb     # proposed method (AgMA-Diff)
├── requirements.txt                         # Python dependencies (PyTorch installed separately)
├── figures/                                 # methodology diagram and result figures
└── README.md
```

---

## Method overview

![AgMA-Diff methodology](figures/methodology_AgMA_Diff.png)

| Stage | What it does |
|---|---|
| **S1: Mode discovery** | Frozen RETFound features → PCA → per-grade k-means (k chosen by silhouette score) → CBLOF flags sparse modes |
| **S2: Mode-conditioned LoRA** | One Stable Diffusion 1.5 LoRA (rank 8) for grades 1–4. The prompt contains the grade and a rare mode token, with 15% token dropout and prior preservation |
| **S3: Three-gate verifier** | **G1** semantic score window calibrated on real validation images · **G2** realism test (k-NN manifold of real images) · **G3** memorisation and duplicate check |
| **S5: Augmentation agent** | Observe → Plan (Qwen2.5-1.5B-Instruct or UCB bandit) → Act (generate + verify) → Reflect (per-arm commit/rollback with a QWK constraint) |
| **S4: Downstream classifier** | LANet (VGG-16-BN) trained on real images plus the accepted synthetic images, using the same protocol as the base paper |

---

## Results: reproduction of the base paper (DDR test set, LANet-VGG)

| Configuration | Paper BAcc | Paper QWK | Paper Acc | **Ours BAcc** | **Ours QWK** | **Ours Acc** |
|---|---:|---:|---:|---:|---:|---:|
| LANet (real only) | 66.84 | 85.06 | 83.67 | **68.40** | **83.62** | **81.03** |
| + oversampling | 68.78 | 82.47 | 79.06 | **71.43** | **79.84** | **72.71** |
| + ECC DM w/o filter | 74.20 | 81.38 | 75.02 | **70.88** | **80.50** | **74.86** |
| + ECC DM w/ filter | 73.82 | 82.58 | 76.80 | **69.86*** | **80.14*** | **76.16*** |

\* Only 299 candidates passed the [0.7, 0.9] filter instead of 2,000.

**Per-grade recall (%)**

| Configuration | No DR | Mild | Moderate | Severe | PDR | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| LANet | 98.88 | 43.39 | 62.20 | 53.52 | 84.00 | 67.07 |
| + oversampling | 93.67 | 78.84 | 42.93 | 70.42 | 71.27 | 60.87 |
| + ECC DM w/o filter | 97.07 | 60.85 | 45.46 | 74.65 | 76.36 | 60.97 |
| + ECC DM w/ filter | 97.34 | 69.84 | 47.92 | 54.93 | 79.27 | 64.53 |

**Key findings**
- Diffusion augmentation raises severe-grade recall from 53.5% to 74.7%, but lowers QWK by 3.1 points and accuracy by 6.2 points, because moderate-DR recall drops by 14–19 points.
- The confidence-based filter accepts only 3.5–6.4% of the candidates, and it admits some visually implausible images.

AgMA-Diff results will be added after the final runs and ablations are complete.

### Differences from the paper's setup

The reproduction runs on a single GPU with a reduced schedule. Every difference is listed below.

| Setting | Paper | This repo |
|---|---|---|
| Effective batch (LoRA) | 96 | 16 |
| LoRA steps per grade | 1,000–1,500 | 600 |
| Synthetic images | 2K + 2K | 1K + 1K |
| Classifier epochs | 100 | 20 |
| Semantic scorer | 6 LANet models | 1 reproduced LANet-VGG |
| Prior pool | DDR, APTOS, EyePACS | DDR, APTOS |

The base paper's six scorer checkpoints are not included in the public checkpoint archive. AgMA-Diff therefore uses the same scorer as the reproduction, so the comparison stays fair.

---

## Requirements

- Linux with an NVIDIA GPU with **≥ 20 GB VRAM**. Developed on an RTX 4000 Ada (20 GB).
- Python **3.12** (3.11 also works; 3.10 does not).
- A Hugging Face account with access to:
  - [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
  - [`YukunZhou/RETFound_mae_natureCFP`](https://huggingface.co/YukunZhou/RETFound_mae_natureCFP) (gated: click *Agree and access repository*)
  - [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

### Environment setup

```bash
git clone https://github.com/<your-username>/AgMA-Diff.git
cd AgMA-Diff

# Python 3.12 via conda (no sudo needed), then a venv on top of it
conda create -y -p ./py312 python=3.12
./py312/bin/python -m venv sara_genai
source sara_genai/bin/activate

python -m pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Jupyter kernel
python -m ipykernel install --user --name sara_genai --display-name "Python (sara_genai)"

# Hugging Face login (token with read access)
hf auth login
```

---

## Datasets

| Dataset | Use | Source |
|---|---|---|
| **DDR** (DR_grading) | Training, validation and test (the only evaluation dataset) | [Hugging Face: ctmedtech/DDR-dataset](https://huggingface.co/datasets/ctmedtech/DDR-dataset) |
| **APTOS 2019** | Prior-preservation images for diffusion fine-tuning only | [Kaggle: aptos2019-blindness-detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) |

Download DDR (about 14 GB):
```bash
export HF_HUB_DISABLE_XET=1
hf download ctmedtech/DDR-dataset --repo-type dataset \
   --include "DR_grading/*" --local-dir data/raw/DDR --max-workers 4
```

Expected folder layout:
```
AgMA-Diff/
└── data/
    ├── raw/DDR/DR_grading/
    │   ├── train/  valid/  test/
    │   └── train.txt  valid.txt  test.txt
    └── aptos2019-blindness-detection/
        ├── train_images/  test_images/
        └── train.csv  test.csv  sample_submission.csv
```

---

## How to run

1. In **cell 0.2** of both notebooks, set the project path:
   ```python
   PROJECT_ROOT = Path("/path/to/AgMA-Diff")
   ```
2. **Run notebook 1** (`01_base_paper_reproduction_local.ipynb`) first, with `PROFILE = "fast"`. It:
   - preprocesses DDR and builds the paper's split (train 6,260; val 47 per grade; test 3,759)
   - trains the ECC diffusion LoRAs and generates and filters candidates
   - trains LANet for the four Table-1 configurations
   - writes `outputs/base_repro/results/table1_reproduction.csv`
3. **Run notebook 2** (`02_AgMA_Diff_full_method_local.ipynb`) afterwards. It reuses notebook 1's split and scorer and writes the comparison table to `outputs/agma/results/comparison_table.csv`.
4. **Ablations:** in cell 0.2 of notebook 2, change `VARIANT_NAME` to one of the following, then restart and run all:

   | `VARIANT_NAME` | What it tests |
   |---|---|
   | `"full"` | the complete method |
   | `"bandit_agent"` | UCB planner instead of the LLM |
   | `"no_agent_fixed"` | base-paper recipe, no agent |
   | `"no_mode_tokens"` | grade-only prompts |
   | `"G1_only"` | semantic gate only |

Each notebook has a **pre-flight cell** (2b in notebook 1, 1b in notebook 2). It checks the data, model access, disk space and GPU memory before training starts. Every stage saves checkpoints, so a run can be resumed after an interruption by running all cells again.

**Approximate runtime on one RTX 4000 Ada ("fast" profile):**
- notebook 1: about 9–12 h
- notebook 2: about 5–6 h for the full variant

---

## Outputs

```
outputs/
├── base_repro/            # notebook 1: splits, diffusion LoRAs, synthetic images, classifiers, results, figures
└── agma/                  # notebook 2: features, mode bank, joint LoRA, agent rounds, classifiers, results, figures
```

---

## Acknowledgements and citation

The reproduction is based on the official code of the base paper:
<https://github.com/AlanZhang1995/ECC_DM_for_DR>

```bibtex
@inproceedings{zhang2025ecc,
  author    = {Zhang, Haochen and Heinke, Anna and Nagel, Ines D. and Bartsch, Dirk-Uwe G. and
               Freeman, William R. and Nguyen, Truong Q. and An, Cheolhong},
  title     = {Class-Conditioned Image Synthesis with Diffusion for Imbalanced Diabetic Retinopathy Grading},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
  series    = {LNCS},
  volume    = {15963},
  pages     = {56--66},
  year      = {2025}
}
```

We also use RETFound (Zhou et al., *Nature* 2023), Stable Diffusion 1.5, LANet (Xia et al., 2023), the DDR dataset (Li et al., *Information Sciences* 2019) and APTOS 2019. Please cite the original works if you use them.

This repository was developed for the course *CS6055 Research Topics in Generative AI* at FAST-NUCES, Islamabad.
