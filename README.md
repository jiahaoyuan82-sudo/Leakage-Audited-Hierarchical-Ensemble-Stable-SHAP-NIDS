# Leakage-Audited Hierarchical Ensemble Learning with Stable SHAP Explanations for Network Intrusion Detection

This repository contains the code needed to rerun the experiment described in *Leakage-Audited Hierarchical Ensemble Learning with Stable SHAP Explanations for Network Intrusion Detection*. It contains **no dataset, trained models, saved predictions, or result figures**. The two curated UNSW-NB15 CSVs must be supplied by the person running it. The data were collected in the UNSW cyber range and combine normal activity with generated attacks; they are not real enterprise production traffic.

## What this code runs

The pipeline audits the CSVs, splits the official training table into a seed-42 stratified training/validation pair, fits train-only preprocessing, and trains Logistic Regression, Random Forest, Extra Trees, XGBoost, LightGBM, a direct ten-class LightGBM, and two-stage ensembles. It selects stage weights and the binary gate threshold using validation only, evaluates the official test once, performs 300 paired bootstrap resamples, fits class-weight ablations and two auxiliary seeds, computes SHAP for **XGBoost and LightGBM only** using three stratified bootstrap samples of at most 100 rows per model and stage, and retrains a compact 20-field ensemble from the stable SHAP ranking. Only each sampled SHAP slice is converted to dense. Extra Trees is not explained by SHAP; the weighted SHAP consensus is not the exact SHAP of the complete ensemble.

The final two scripts independently reload and evaluate the saved models, audit exact 42-predictor train/test overlaps, and recompute Log Loss from named probability columns. Generated predictions, models, result tables, and research plots are written locally to ignored folders.

## Requirements

- Python 3.13 is recommended. The recorded run used Python 3.13 on Windows 11. Other environments may produce small numeric or timing differences.
- A machine with approximately 8 CPU cores and 16 GB RAM is a practical target. The complete training run can take several hours; there is no automatic four-hour cutoff.
- Install the exact recorded package versions in `requirements.txt`. Internet access may be needed for installation and for obtaining the CSVs; the experiment itself does not download data.

Create an environment from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, activate with `source .venv/bin/activate` instead.

## Obtain the two curated CSVs

The original source is the [UNSW-NB15 dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset). Use its curated training and testing CSVs, **not** the raw PCAP files. The official endpoint required authentication during the recorded experiment, so the exact files used for the reported run were obtained from these public mirror URLs:

- [UNSW_NB15_training-set.csv](https://raw.githubusercontent.com/DSpike/Block_fed/master/UNSW_NB15_training-set.csv)
- [UNSW_NB15_testing-set.csv](https://raw.githubusercontent.com/DSpike/Block_fed/master/UNSW_NB15_testing-set.csv)

By default, put them here, retaining their filenames:

```text
data/raw/UNSW_NB15_training-set.csv
data/raw/UNSW_NB15_testing-set.csv
```

The runner checks both SHA-256 digests and the 45-column schema before training:

| File | Rows | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| Training | 175,341 | 32,117,676 | `0ebd9eb06d53ae223fbe7948737985e851e00e9c20140508d653de4b1655fdd6` |
| Testing | 82,332 | 15,298,467 | `7ec02e7e44d72bd265716b33fda0c7f2188b658e3a4ae1aa2c4b306134cd818c` |

If a file differs, the runner stops. Do not silently substitute a different split when comparing with the reported results.
You can also keep the CSVs outside the repository and pass their containing directory with `--data-dir`; this keeps the code folder free of raw data even while running.

## Run

From the repository root, first check the files and installed packages without fitting models:

```powershell
python src/reproduce_all.py --check
```

Then run the entire pipeline with one command:

```powershell
python src/reproduce_all.py
```

If the CSVs are stored elsewhere, use `python src/reproduce_all.py --data-dir "C:\path\to\csv-folder"` (and the same option with `--check`).

The command runs the scripts in this order:

1. `run_experiment.py`: data audit, all training/validation models, official-test predictions, bootstrap, SHAP, compact refit, and raw figures.
2. `complete_evaluation.py`: reload full and compact models, recompute compact predictions and metrics, and check 128-row probability shapes and finiteness.
3. `audit_feature_overlap_logloss.py`: exact CSV-row and 42-predictor overlap audit, plus named-column Log Loss recalculation.
4. `audit_predictor_overlap_sensitivity.py`: post-hoc test scoring with exact 42-predictor matches excluded, using saved predictions without retraining.

The runner then checks SHAP scope, the 300 bootstrap rows, and Log Loss agreement, writing `results/reproduction_check.json`. Outputs are created in `data/processed/`, `models/`, `results/`, `figures/`, and `logs/`; all are excluded by `.gitignore`.

## Interpreting the output

- The official 82,332-row test set is the primary evaluation. The overlap sensitivity subset is a **post-hoc diagnostic**, not another tuning set.
- Complete 45-column CSV rows have zero cross-split matches, but the recorded data have 8,541 test rows with an exact match on all 42 model input fields in training. This limitation must accompany any performance claim.
- Log Loss is calculated by matching each true class to its **named** probability column, normalizing each row, and clipping true-class probability below the float64 machine epsilon before taking `-log(p)`. This avoids the old column-order mismatch.
- The publication artwork was prepared separately. The `figures/` directory generated by this code contains the experiment plots, not the manually designed journal infographics.
- The saved direct LightGBM bundle used an upper limit of **700** estimators and early-stopped at iteration **366**. The manuscript parameter table currently says **650**. This repository retains the value confirmed in the saved fitted model. Because early stopping occurred before either cap, this discrepancy does not change the recorded fitted iteration, but the manuscript table should be corrected before claiming exact parameter agreement.
- This repository was cleaned and checked without rerunning the multi-hour training pipeline during packaging. Syntax, imports, CLI behavior, input hashes, and metric-order logic were checked; a fresh end-to-end run on another machine remains to be performed by the user.

## Repository contents

```text
README.md                 usage, data provenance, limitations
requirements.txt          exact recorded Python packages
CITATIONS.md              dataset and method attribution
.gitignore                excludes data and generated artifacts
data/raw/.gitkeep         empty placeholder; add CSVs locally
src/reproduce_all.py      one-command orchestration and verification
src/run_experiment.py     training, evaluation, SHAP, bootstrap
src/complete_evaluation.py
src/audit_feature_overlap_logloss.py
src/audit_predictor_overlap_sensitivity.py
src/scoring.py            named-probability Log Loss
```

No license has been chosen for this code. Add a license that matches your intended GitHub sharing terms before inviting reuse.
