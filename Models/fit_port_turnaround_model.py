from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.customer_paths import (  # noqa: E402
    CustomerPaths,
    describe_customers,
    ensure_customer_dirs,
    resolve_customer,
)

ML_SUBDIR = "ML"
ARTIFACT_DIR = ROOT / "Models" / "Artifacts"
TARGET_COLUMN = "DAYS_IN_PORT"


@dataclass
class ModelPaths:
    artifact: Path
    metrics_json: Path
    report_txt: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline Decision Tree model for port turnaround duration.")
    parser.add_argument(
        "--customer",
        "-c",
        help="Customer slug (default: northernlights). Use --list-customers to see options.",
    )
    parser.add_argument(
        "--list-customers",
        action="store_true",
        help="Print available customer identifiers and exit.",
    )
    parser.add_argument("--max-depth", type=int, default=8, help="Decision tree max depth.")
    parser.add_argument("--min-samples-leaf", type=int, default=20, help="Minimum samples per leaf.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument(
        "--use-log-target",
        action="store_true",
        help="Train on log-transformed target (LOG_DAYS_IN_PORT) and invert predictions back to days.",
    )
    return parser.parse_args()


def configure_customer(slug: str | None) -> CustomerPaths:
    paths = resolve_customer(slug)
    ensure_customer_dirs(paths)
    (paths.derived_dir / ML_SUBDIR).mkdir(parents=True, exist_ok=True)
    (paths.derived_dir / "QA" / "ml").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / paths.key).mkdir(parents=True, exist_ok=True)
    return paths


def load_metadata(paths: CustomerPaths) -> dict:
    metadata_path = paths.derived_dir / ML_SUBDIR / "port_turnaround_features.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Feature metadata missing for customer '{paths.key}'. "
            "Run Models/train_port_turnaround_model.py first."
        )
    with metadata_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected split file missing: {path}")
    return pd.read_parquet(path)


def build_pipeline(
    categorical_features: list[str],
    numeric_features: list[str],
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> Pipeline:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1.0, y_true)))
    return {"mae": float(mae), "rmse": float(rmse), "smape": float(mape)}


def format_report(customer: str, metrics: dict, baseline: dict) -> str:
    lines = [
        f"Port turnaround ML model report ({customer})",
        "",
        "Model metrics:",
        f"  Train MAE: {metrics['train']['mae']:.3f}",
        f"  Train RMSE: {metrics['train']['rmse']:.3f}",
        f"  Train sMAPE: {metrics['train']['smape']:.3f}",
        "",
        f"  Validation MAE: {metrics['validation']['mae']:.3f}",
        f"  Validation RMSE: {metrics['validation']['rmse']:.3f}",
        f"  Validation sMAPE: {metrics['validation']['smape']:.3f}",
        "",
        "Baseline (training median) comparison:",
        f"  Validation MAE: {baseline['validation']['mae']:.3f}",
        f"  Validation RMSE: {baseline['validation']['rmse']:.3f}",
        f"  Validation sMAPE: {baseline['validation']['smape']:.3f}",
    ]
    return "\n".join(lines)


def determine_paths(paths: CustomerPaths) -> ModelPaths:
    artifact = ARTIFACT_DIR / paths.key / "port_turnaround_dt.joblib"
    metrics_json = ARTIFACT_DIR / paths.key / "port_turnaround_dt_metrics.json"
    report_txt = paths.derived_dir / "QA" / "ml" / "port_turnaround_model_report.txt"
    return ModelPaths(artifact=artifact, metrics_json=metrics_json, report_txt=report_txt)


def main() -> None:
    args = parse_args()
    if args.list_customers:
        print(describe_customers())
        return

    paths = configure_customer(args.customer)
    metadata = load_metadata(paths)

    train_path = ROOT / metadata["train_path"]
    val_path = ROOT / metadata["validation_path"]
    train_df = load_split(train_path)
    val_df = load_split(val_path)

    categorical_features = metadata["categorical_features"] + metadata.get("derived_categorical_features", [])
    numeric_features = metadata["numeric_features"]

    feature_columns = categorical_features + numeric_features
    missing_features = [col for col in feature_columns + [TARGET_COLUMN] if col not in train_df.columns]
    if missing_features:
        raise ValueError(f"Training dataset missing columns: {', '.join(missing_features)}")

    target_col = metadata.get("log_target_column") if args.use_log_target else TARGET_COLUMN
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' missing from training data.")

    X_train = train_df[feature_columns].copy()
    X_val = val_df[feature_columns].copy()

    for col in categorical_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("string").fillna("MISSING")
            X_val[col] = X_val[col].astype("string").fillna("MISSING")

    for col in numeric_features:
        if col in X_train.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_val[col] = pd.to_numeric(X_val[col], errors="coerce")

    y_train = train_df[target_col].to_numpy()
    y_val = val_df[target_col].to_numpy()

    pipeline = build_pipeline(
        categorical_features,
        numeric_features,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)

    if args.use_log_target:
        train_eval_true = np.expm1(y_train)
        train_eval_pred = np.expm1(train_pred)
        val_eval_true = np.expm1(y_val)
        val_eval_pred = np.expm1(val_pred)
    else:
        train_eval_true = y_train
        train_eval_pred = train_pred
        val_eval_true = y_val
        val_eval_pred = val_pred

    metrics = {
        "train": compute_metrics(train_eval_true, train_eval_pred),
        "validation": compute_metrics(val_eval_true, val_eval_pred),
    }

    baseline_pred_train = np.full_like(train_eval_true, np.median(train_eval_true), dtype=float)
    baseline_pred_val = np.full_like(val_eval_true, np.median(train_eval_true), dtype=float)
    baseline_metrics = {
        "validation": compute_metrics(val_eval_true, baseline_pred_val),
        "train": compute_metrics(train_eval_true, baseline_pred_train),
    }

    model_paths = determine_paths(paths)
    joblib.dump(
        {
            "pipeline": pipeline,
            "target_column": TARGET_COLUMN,
            "use_log_target": args.use_log_target,
            "feature_columns": feature_columns,
            "metadata": metadata,
        },
        model_paths.artifact,
    )

    metrics_payload = {
        "metrics": metrics,
        "baseline": baseline_metrics,
        "parameters": {
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": args.random_state,
            "use_log_target": args.use_log_target,
        },
    }
    model_paths.metrics_json.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    model_paths.report_txt.write_text(
        format_report(paths.key, metrics, baseline_metrics),
        encoding="utf-8",
    )

    print(f"[{paths.key}] Trained DecisionTreeRegressor")
    print(f"  Artifact: {model_paths.artifact.relative_to(ROOT)}")
    print(f"  Metrics:  {model_paths.metrics_json.relative_to(ROOT)}")
    print(f"  Report:   {model_paths.report_txt.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
