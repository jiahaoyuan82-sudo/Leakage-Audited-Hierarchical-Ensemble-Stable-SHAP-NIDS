"""Reproduce the UNSW-NB15 experiment from the two curated CSV files.

No dataset, trained model, or prediction file is bundled with this repository.
Run this script from any working directory after placing the CSVs in data/raw.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw"
EXPECTED = {
    "UNSW_NB15_training-set.csv": "0ebd9eb06d53ae223fbe7948737985e851e00e9c20140508d653de4b1655fdd6",
    "UNSW_NB15_testing-set.csv": "7ec02e7e44d72bd265716b33fda0c7f2188b658e3a4ae1aa2c4b306134cd818c",
}
SCRIPTS = (
    "run_experiment.py",
    "complete_evaluation.py",
    "audit_feature_overlap_logloss.py",
    "audit_predictor_overlap_sensitivity.py",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_inputs(raw_dir):
    for name, expected_hash in EXPECTED.items():
        path = raw_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Download the curated CSV from the UNSW-NB15 "
                "dataset page, then place it in data/raw. See README.md."
            )
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {name}: {actual_hash}. This code targets "
                "the exact CSV files used in the reported experiment."
            )
        with path.open(newline="", encoding="utf-8-sig") as stream:
            columns = next(csv.reader(stream))
        if len(columns) != 45 or not {"id", "label", "attack_cat"}.issubset(columns):
            raise ValueError(f"Unexpected 45-column schema in {path}")
    print("Input CSV hashes and schemas match the recorded experiment.", flush=True)


def check_dependencies():
    missing = []
    mismatched = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        package, expected = line.split("==", 1)
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
            continue
        if actual != expected:
            mismatched.append(f"{package}: installed {actual}, recorded {expected}")
    if missing:
        raise RuntimeError(
            "Missing packages: " + ", ".join(missing) + ". Run pip install -r requirements.txt."
        )
    if mismatched:
        print("Version differences may change numerical results:", *mismatched, sep="\n  ", flush=True)
    else:
        print("Dependency versions match requirements.txt.", flush=True)


def check_outputs():
    results = ROOT / "results"
    summary = json.loads((results / "summary_metrics.json").read_text(encoding="utf-8"))
    selected = json.loads((results / "stable_feature_selection.json").read_text(encoding="utf-8"))
    loss = json.loads((results / "probability_logloss_reaudit.json").read_text(encoding="utf-8"))
    overlap = json.loads((results / "feature_overlap_audit.json").read_text(encoding="utf-8"))
    if selected.get("test_data_used_for_selection") is not False:
        raise RuntimeError("SHAP feature-selection scope is not recorded as train-only")
    if selected.get("explained_models") != ["XGBoost", "LightGBM"]:
        raise RuntimeError("Unexpected SHAP model scope")
    full_reported = summary["test_multiclass_hierarchical"]["Log_Loss"]
    full_recomputed = loss["full_prediction_file"]["by_model"]["full_hierarchical_ensemble"]["mean_log_loss_named_columns"]
    if abs(full_reported - full_recomputed) > 1e-9:
        raise RuntimeError("Saved named-column Log Loss disagrees with the summary")
    with (results / "paired_bootstrap_300.csv").open(newline="", encoding="utf-8") as stream:
        count = sum(1 for _ in csv.DictReader(stream))
    if count != 300:
        raise RuntimeError(f"Expected 300 paired bootstrap rows, found {count}")
    report = {
        "input_sha256": EXPECTED,
        "full_macro_f1": summary["test_multiclass_hierarchical"]["Macro_F1"],
        "full_mcc": summary["test_multiclass_hierarchical"]["MCC"],
        "full_log_loss_named_columns": full_recomputed,
        "cross_split_42_feature_matched_test_rows": overlap["cross_split_feature_affected_test_rows"],
        "paired_bootstrap_replicates": count,
        "shap_test_used_for_feature_selection": False,
    }
    (results / "reproduction_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check CSV hashes and package versions without training")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_RAW,
                        help="directory containing the two CSVs; default: data/raw")
    args = parser.parse_args()
    raw_dir = args.data_dir.expanduser().resolve()
    check_inputs(raw_dir)
    check_dependencies()
    if args.check:
        return
    for script in SCRIPTS:
        print(f"Running {script}", flush=True)
        env = os.environ.copy()
        env["UNSW_RAW_DATA_DIR"] = str(raw_dir)
        subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, env=env, check=True)
    check_outputs()


if __name__ == "__main__":
    main()
