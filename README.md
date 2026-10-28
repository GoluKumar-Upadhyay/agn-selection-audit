<div align="center">

# AGN Selection Audit

### Selection-aware evaluation of image–tabular classification of AGN candidates in SDSS

*Shortcut audit · matched cohorts · corrected multi-split statistics · cross-task transfer*

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B0-ee4c2c)
![Data](https://img.shields.io/badge/data-SDSS%20DR17-informational)
![Status](https://img.shields.io/badge/status-manuscript%20in%20preparation-orange)

</div>

---

## Overview

Machine-learning classifiers for active galactic nuclei (AGN) are usually judged by random-holdout accuracy. On SDSS, that number can be inflated by *how the labelled sample was built*. This repository contains the code, seeds and result tables behind a **selection-aware evaluation** of image + catalogue (tabular) classification, applied to two tasks:

| Task | Labels | Objects | Images |
|---|---|---|---|
| **QSO vs galaxy** | SDSS spectroscopic class | 14,933 (3,686 in the matched cohort) | 64 × 64 px, 0.4″/px (25.6″ field) |
| **BPT AGN vs star-forming** | MPA-JHU BPT class, matched in redshift and magnitude | 4,276 (2,138 per class) | 128 × 128 px, 0.2″/px (25.6″ field) |

The project does **not** propose a new architecture. It asks what a conventional random split hides, and answers with an audit protocol, matched cohorts, corrected inference for overlapping splits, and an external transfer test.

## Key findings

| Finding | Result |
|---|---|
| Redshift alone separates the QSO sample | AUC **0.9974**; only 24 of 4,958 QSOs lie inside the galaxy redshift range (galaxies z ≤ 0.030) |
| The sample is a non-random database slice | Sky position alone AUC 0.63; object ID alone 0.62 |
| Shortcut variables matter | XGBoost 0.9985 → **0.9886** on the matched cohort when redshift and the redshift-derived M_r are removed |
| Classical rules collapse on matched data | Colour cut AUC 0.566, morphology rule 0.663 |
| Images + catalogue vs images (frozen encoder) | +0.019 AUC (QSO) and +0.063 (BPT), robust after Holm correction |
| Fine-tuned encoder | Image-only 0.9847, indistinguishable from XGBoost (0.9886); concatenation 0.9936 vs XGBoost +0.0050 (Holm p = 0.033 in-family, ≈ 0.055 pooled) |
| Gated attention vs concatenation | No reliable benefit in either task (CIs include zero) |
| QSO-trained model on BPT AGN | AUC **0.557** (no transfer) |
| Matched BPT task | Redshift alone 0.514; XGBoost 0.922; image-only 0.867; concatenation 0.928; gated 0.931 |

> All values are means over ten random 70/15/15 splits. Confidence intervals and p-values use a corrected resampled *t*-test (Nadeau & Bengio) with Holm adjustment, because the splits overlap. See the manuscript for details and caveats.

## Repository layout

```
agn_selection_audit/
├── data/
│   ├── sdss_agn_dataset.csv          # SDSS table (14,933 objects after cleaning)
│   ├── image_cohort.csv              # created by extract_features.py
│   ├── effb0.pth                     # EfficientNet-B0 weights (download, see below)
│   └── images/{agn,galaxy}/          # QSO-task cutouts (download, see below)
├── src/                              # all code (see script map below)
├── notebooks/
│   └── AGN_finetune_colab.ipynb      # GPU fine-tuning on Google Colab
├── results/                          # tables, predictions, logs (generated)
├── figures/                          # figures (generated)
├── download_sdss_images.py           # cutout downloader for the QSO task
└── requirements.txt
```

## Quick start

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Developed and tested on Python 3.12 (Linux); the full pipeline was also run on Python 3.14 (Windows, CPU). If PyTorch has no wheel for your Python version, use 3.12.

On **Windows PowerShell**, set UTF-8 once per session so that tables with special characters are written correctly:

```powershell
$env:PYTHONUTF8="1"
```

### 2. Data

| Asset | How to get it |
|---|---|
| `data/sdss_agn_dataset.csv` | Included in the repository |
| `data/images/{agn,galaxy}/` | `python download_sdss_images.py` (SDSS cutout service; 64 px, 0.4″/px, no contrast stretch) |
| `data/effb0.pth` | [EfficientNet-PyTorch release 1.0, `efficientnet-b0-355c32eb.pth`](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth) |

### 3. Smoke test (about 1 minute)

```bash
python src/audit_tabular.py
```

This runs the tabular audit and checks that the environment works.

## Reproducing the results

Run from the repository root. Times are approximate, as measured on a laptop CPU.

| Step | Command | Time | Produces |
|---|---|---|---|
| 1. Selection audit | `python src/audit_tabular.py` | ~1 min | Audit tables, SHAP, negative controls |
| 2. Image features | `python src/extract_features.py` | ~10 min | `data/effb0_tokens.npy`, `data/image_cohort.csv` |
| 3. Matched-cohort experiments | `python src/run_paper_experiments.py` | ~2 h | `results/paper/predictions.pkl` |
| 4. Tables | `python src/make_paper_tables.py` | ~1 min | `results/paper_tables/Table*.csv`, `figures/paper/` |
| 5. Corrected statistics | `python src/stats_corrected.py` | seconds | Corrected paired tests and CIs |
| 6. Tuned baselines | `python src/tune_and_test.py` | ~10 min | Default vs tuned baselines |
| 7. Clean figures | `python src/make_clean_figures.py qso` | seconds | ROC / PR / confusion / reliability |

<details>
<summary><b>BPT task (narrow-line AGN vs star-forming)</b></summary>

```bash
python src/external_bpt.py fetch            # SDSS SkyServer queries (needs internet)
python src/external_bpt.py score            # transfer of the QSO-trained model (AUC ~0.557)
python src/bpt_task.py                      # catalogue-only matched BPT task
python src/download_bpt_images.py           # 128 px cutouts, resumable
python src/extract_features_bpt.py
python src/run_bpt_fusion.py                # ~30 min, saves results/paper/bpt_predictions.pkl
python src/make_clean_figures.py bpt
python src/stats_corrected.py               # now also covers the BPT task
```

Before the first `fetch`, verify the table and column names (`galSpecExtra.bptclass`, codes 1 = star forming, 4 = AGN) in the SkyServer schema browser.
</details>

<details>
<summary><b>Fine-tuned EfficientNet-B0 (GPU)</b></summary>

Fine-tuning takes about 1.5 hours per run on a CPU, so use a GPU (the notebook targets a free Colab T4):

1. Build `colab_data.zip` (see the first cell of `notebooks/AGN_finetune_colab.ipynb`).
2. Run the notebook: 10 seeds × 3 modes (`img_gap`, `concat`, `attn_gate`).
3. Unzip the downloaded predictions into `results/`, then:

```bash
python src/analyze_finetune.py
```

Single run:

```bash
python src/train_finetune.py --task qso --mode attn_gate --tab photo12 --seed 42
```
</details>

## Script map

| Script | Purpose |
|---|---|
| `data.py` | Loading, cleaning, 14 features, feature groups, stratified 70/15/15 splits |
| `stats.py` | DeLong test, bootstrap CIs, ECE, conformal utilities |
| `stats_corrected.py` | Corrected resampled *t*-test, Holm adjustment, effect sizes |
| `audit_tabular.py` | Selection audit: overlap, shortcut-only baselines, peeling, negative controls, SHAP |
| `extract_features.py` | Frozen EfficientNet-B0 tokens (7 × 7 × 1280) for the QSO task |
| `models_fusion.py` | Image-only, tabular-only, concatenation, cross-attention (with and without gate) |
| `run_paper_experiments.py` | All models on the matched cohort, 10 seeds |
| `make_paper_tables.py` | Result tables (CSV and Markdown, numbered by this script, not by the manuscript) and the first figure set |
| `make_clean_figures.py` | Averaged ROC / PR, confusion matrices, reliability diagrams |
| `tune_and_test.py` | Tuned XGBoost and MLP baselines and paired tests |
| `external_bpt.py` | BPT cohort query, scoring of the QSO-trained model |
| `bpt_task.py`, `download_bpt_images.py`, `extract_features_bpt.py`, `run_bpt_fusion.py` | Matched BPT task |
| `train_finetune.py`, `analyze_finetune.py` | End-to-end EfficientNet-B0 and its statistics |
| `external_validation.py`, `external_figures.py` | Metrics and figures for an external cohort with negative controls |
| `run_fusion.py`, `analyze_fusion.py` | Earlier pilot pipeline (2,220 objects); superseded by `run_paper_experiments.py` |
| `fetch_overlap_cohort.py` | Draft query for a redshift-overlapping cohort; **not yet run or validated** |

## Method in brief

- **Splits.** Stratified 70/15/15 with scikit-learn `train_test_split`, `random_state` = seed (42–51 for image and fusion models; 42–46 for the tabular audit). Every model in a seed uses identical train, validation and test objects.
- **Matching.** Coarsened exact matching with a fixed generator (seed 0): r in 0.5 mag bins and log₁₀ R50 in 0.1 dex bins (QSO task).
- **Features.** 14 catalogue features; a 12-feature shortcut-free set omits redshift and the redshift-derived absolute magnitude.
- **Image models.** EfficientNet-B0 with frozen ImageNet features (PCA to 128-d tokens) or fine-tuned end to end (two phases, flips and 90° rotations).
- **Fusion.** Concatenation, and tabular-query cross-attention over 49 image tokens with a learned gate.
- **Statistics.** Paired DeLong per split; across splits a corrected resampled *t*-test with variance factor (1/k + n_test/n_train), Cohen's *d*_z, and Holm adjustment. Uncorrected Wilcoxon p-values are descriptive only.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: effb0_tokens.npy` | Run `python src/extract_features.py` first |
| `FileNotFoundError: data/images/agn` | Run `download_sdss_images.py` and check that `agn` and `galaxy` contain `.jpg` files |
| `UnicodeEncodeError` on Windows | `$env:PYTHONUTF8="1"` |
| `ModuleNotFoundError: tabulate` | `pip install tabulate` (listed in `requirements.txt`) |
| `can't find '__main__' module` | A stray space, e.g. `src/ script.py`; write `src/script.py` |
| No PyTorch wheel for your Python | Use Python 3.12 |

## Limitations

- The QSO-versus-galaxy sample is a **non-random database slice** with almost no redshift overlap between classes; a redshift-matched QSO analysis is not possible.
- Ten random splits are **not independent experiments**; corrected intervals are approximate and no independent held-out test set was available.
- The fine-tuned encoder was run on the QSO task only; the BPT task uses frozen features.
- There is no independent X-ray or mid-infrared validation set yet.
- BPT labels are one definition of AGN and require well-measured emission lines.
- Image features also encode angular size and brightness, so images are not shortcut-free.

## Citation

The accompanying manuscript is in preparation. Citation details (authors, venue, DOI) will be added here after publication.

```bibtex
@misc{agn_selection_audit,
  title  = {Selection-Aware Evaluation of Image--Tabular Classification of AGN Candidates in SDSS},
  note   = {Manuscript in preparation; citation details to be added}
}
```

## Acknowledgements and data terms

- Data: [Sloan Digital Sky Survey](https://www.sdss.org/) DR17 and the MPA-JHU emission-line catalogue. Please follow the SDSS data-use and acknowledgement policies.
- EfficientNet-B0 ImageNet weights: [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) release 1.0.

## License and contact

License: *to be chosen by the authors before release* (add a `LICENSE` file).
Contact: *authors' details to be added.*
