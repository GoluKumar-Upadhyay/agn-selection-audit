# AGN/QSO selection-shortcut audit + corrected image–tabular fusion

## Run order
1. `python src/audit_tabular.py`     – classical baselines, feature ablations, shortcut peeling, negative controls, SHAP, calibration, conformal (CPU, ~1 min)
2. `python src/extract_features.py`  – frozen EfficientNet-B0 tokens (needs data/effb0.pth = github.com/lukemelas/EfficientNet-PyTorch release 1.0 efficientnet-b0-355c32eb.pth, and data/images/{agn,galaxy}/)
3. `SEEDS=42,...,51 python src/run_fusion.py` then `python src/analyze_fusion.py` – 13 variants x 10 splits, paired DeLong, modality perturbation (CPU, ~20 min)
4. GPU: `python src/train_finetune.py --mode attn_gate --tab photo12 --seed 42` (loop over seeds/modes)
5. New data: `python src/fetch_overlap_cohort.py` ; external test: `python src/external_validation.py --cohort ...`

## Verification status (be honest in the paper)
Executed and producing the shipped results: data.py, stats.py, audit_tabular.py, extract_features.py, run_fusion.py, analyze_fusion.py.
Smoke-tested only (2 epochs, CPU): train_finetune.py.  Self-tested on synthetic data: external_validation.py.
NOT executed (no network in authoring sandbox): fetch_overlap_cohort.py – verify SkyServer table/column names and BPT class codes first.

## Known limitations of the shipped results
Frozen backbone + PCA-128 tokens; ~333 test objects per split; test sets overlap across seeds (across-seed Wilcoxon p-values are descriptive);
EfficientNet weights are the lukemelas port, not torchvision's.

## Paper tables/figures pipeline (added)
1. `python src/run_paper_experiments.py`  (10 seeds, ~20 min CPU; QUICK=1 SEEDS=42 for a smoke run) -> results/paper/
2. `python src/make_paper_tables.py`      -> results/paper_tables/Table{1,2,3,4,5,7}.csv/.md and figures/paper/fig1-7
3. Table 6 / Fig 8 need an external cohort: put scores in results/external_scores.csv (objid,label,subtype,score) and run
   `python src/external_validation.py --cohort results/external_scores.csv` then `python src/external_figures.py`.
Not feasible on the current sample (documented in Table 3): redshift-matched cohort (4 QSOs inside the galaxy z range) and z>0.5 extrapolation (no objects).
Grad-CAM (Fig 7 column) requires fine-tuned checkpoints from train_finetune.py (not implemented in the frozen-feature pipeline).
