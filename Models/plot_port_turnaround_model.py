from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

from utils.customer_paths import (  # noqa: E402
    CustomerPaths,
    describe_customers,
    ensure_customer_dirs,
    resolve_customer,
)

ML_SUBDIR = "ML"
QA_ML_SUBDIR = Path("QA") / "ml"
ARTIFACT_FILENAME = "port_turnaround_dt.joblib"
METRICS_FILENAME = "port_turnaround_dt_metrics.json"
TARGET_COLUMN = "DAYS_IN_PORT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visualisations for the port turnaround ML model.")
    parser.add_argument(
        "--customer",
        "-c",
        help="Customer slug (default northernlights). Use --list-customers to see options.",
    )
    parser.add_argument(
        "--list-customers",
        action="store_true",
        help="List available customers and exit.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
        help="Which split to visualise (default: validation).",
    )
    parser.add_argument(
        "--use-log-target",
        action="store_true",
        help="Interpret model predictions as log1p(DAYS_IN_PORT) and invert to days.",
    )
    return parser.parse_args()


def configure_customer(slug: str | None) -> CustomerPaths:
    paths = resolve_customer(slug)
    ensure_customer_dirs(paths)
    (paths.derived_dir / QA_ML_SUBDIR).mkdir(parents=True, exist_ok=True)
    return paths


def load_metadata(paths: CustomerPaths) -> dict:
    meta_path = paths.derived_dir / ML_SUBDIR / "port_turnaround_features.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata missing for '{paths.key}'. Run Models/train_port_turnaround_model.py first."
        )
    return json.loads(meta_path.read_text())


def load_artifact(paths: CustomerPaths) -> dict:
    artifact_path = ROOT / "Models" / "Artifacts" / paths.key / ARTIFACT_FILENAME
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact missing: {artifact_path}. Train the model via Models/fit_port_turnaround_model.py first."
        )
    return joblib.load(artifact_path)


def load_split(metadata: dict, split: str) -> pd.DataFrame:
    key = "train_path" if split == "train" else "validation_path"
    path = ROOT / metadata[key]
    if not path.exists():
        raise FileNotFoundError(f"Expected split file missing: {path}")
    return pd.read_parquet(path)


def generate_predictions(
    pipeline_bundle: dict,
    metadata: dict,
    df: pd.DataFrame,
    feature_columns: list[str],
    use_log_target: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    pipeline = pipeline_bundle["pipeline"]
    X = df[feature_columns].copy()
    categorical = metadata.get("categorical_features", []) + metadata.get("derived_categorical_features", [])

    for col in categorical:
        if col in X.columns:
            X[col] = X[col].astype("string").fillna("MISSING")
    for col in metadata.get("numeric_features", []):
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    preds = pipeline.predict(X)
    if use_log_target or pipeline_bundle.get("use_log_target"):
        actual = np.expm1(df[pipeline_bundle.get("log_target_column", "LOG_DAYS_IN_PORT")].to_numpy())
        predicted = np.expm1(preds)
    else:
        actual = df[TARGET_COLUMN].to_numpy()
        predicted = preds
    return actual, predicted


def plot_actual_vs_predicted(actual: np.ndarray, predicted: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, alpha=0.3, edgecolors="none")
    max_val = max(actual.max(), predicted.max())
    ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", label="Ideal")
    ax.set_xlabel("Actual DAYS_IN_PORT")
    ax.set_ylabel("Predicted DAYS_IN_PORT")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_residual_hist(actual: np.ndarray, predicted: np.ndarray, out_path: Path, title: str) -> None:
    residuals = predicted - actual
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=30, color="#4c78a8", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (Predicted - Actual) [days]")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.list_customers:
        print(describe_customers())
        return

    paths = configure_customer(args.customer)
    metadata = load_metadata(paths)
    artifact = load_artifact(paths)

    feature_columns = artifact.get("feature_columns") or (
        metadata["categorical_features"]
        + metadata.get("derived_categorical_features", [])
        + metadata["numeric_features"]
    )

    df = load_split(metadata, args.split)
    actual, predicted = generate_predictions(
        artifact,
        metadata,
        df,
        feature_columns,
        use_log_target=args.use_log_target,
    )

    qa_dir = paths.derived_dir / QA_ML_SUBDIR
    title_suffix = "Train" if args.split == "train" else "Validation"
    avp_path = qa_dir / f"port_turnaround_{args.split}_actual_vs_pred.png"
    residual_path = qa_dir / f"port_turnaround_{args.split}_residual_hist.png"

    plot_actual_vs_predicted(actual, predicted, avp_path, f"{title_suffix} Actual vs Predicted")
    plot_residual_hist(actual, predicted, residual_path, f"{title_suffix} Residuals")

    print(f"[{paths.key}] Saved visualisations for {args.split} split:")
    print(f"  - {avp_path.relative_to(ROOT)}")
    print(f"  - {residual_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
