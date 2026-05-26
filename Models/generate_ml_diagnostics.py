from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    metadata_filename: str
    artifact_filename: str
    target_column: str
    split_report_label: str
    output_prefix: str


DATASETS: dict[str, DatasetSpec] = {
    "turnaround": DatasetSpec(
        name="turnaround",
        metadata_filename="port_turnaround_features.json",
        artifact_filename="port_turnaround_dt.joblib",
        target_column="DAYS_IN_PORT",
        split_report_label="port turnaround",
        output_prefix="port_turnaround",
    ),
    "sailing": DatasetSpec(
        name="sailing",
        metadata_filename="sailing_time_features.json",
        artifact_filename="sailing_time_dt.joblib",
        target_column="HOURS_PER_NM",
        split_report_label="sailing time",
        output_prefix="sailing_time",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate permutation-importance diagnostics for saved thesis ML artifacts."
    )
    parser.add_argument(
        "--customer",
        "-c",
        help="Customer slug (default: customer2). Use --list-customers to see options.",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASETS),
        nargs="+",
        default=("turnaround", "sailing"),
        help="Which dataset diagnostics to generate.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
        help="Split used for diagnostics (default: validation).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Number of permutation repeats per feature (default: 20).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top features to show in the saved bar chart (default: 15).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for feature permutation.",
    )
    parser.add_argument(
        "--list-customers",
        action="store_true",
        help="Print available customer identifiers and exit.",
    )
    return parser.parse_args()


def configure_customer(slug: str | None) -> CustomerPaths:
    paths = resolve_customer(slug)
    ensure_customer_dirs(paths)
    (paths.derived_dir / QA_ML_SUBDIR).mkdir(parents=True, exist_ok=True)
    return paths


def load_metadata(paths: CustomerPaths, spec: DatasetSpec) -> dict:
    meta_path = paths.derived_dir / ML_SUBDIR / spec.metadata_filename
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata missing for '{paths.key}' and dataset '{spec.name}': {meta_path}"
        )
    return json.loads(meta_path.read_text())


def load_artifact(paths: CustomerPaths, spec: DatasetSpec) -> dict:
    artifact_path = ROOT / "Models" / "Artifacts" / paths.key / spec.artifact_filename
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Saved artifact missing for '{paths.key}' and dataset '{spec.name}': {artifact_path}"
        )
    return joblib.load(artifact_path)


def load_split(metadata: dict, split: str) -> pd.DataFrame:
    key = "train_path" if split == "train" else "validation_path"
    path = ROOT / metadata[key]
    if not path.exists():
        raise FileNotFoundError(f"Expected split file missing: {path}")
    return pd.read_parquet(path)


def prepare_features(df: pd.DataFrame, metadata: dict, feature_columns: list[str]) -> pd.DataFrame:
    categorical = metadata.get("categorical_features", []) + metadata.get(
        "derived_categorical_features", []
    )
    numeric = metadata.get("numeric_features", []) + metadata.get("derived_numeric_features", [])

    X = df[feature_columns].copy()
    for col in categorical:
        if col in X.columns:
            X[col] = X[col].astype("string").fillna("MISSING")
    for col in numeric:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def predict_in_original_units(payload: dict, X: pd.DataFrame) -> np.ndarray:
    raw_pred = np.asarray(payload["pipeline"].predict(X), dtype=float)
    if payload.get("use_log_target", False):
        return np.expm1(raw_pred)
    return raw_pred


def target_in_original_units(df: pd.DataFrame, payload: dict, spec: DatasetSpec) -> np.ndarray:
    if payload.get("use_log_target", False):
        log_col = payload.get("metadata", {}).get("log_target_column")
        if not log_col:
            log_col = "LOG_DAYS_IN_PORT" if spec.name == "turnaround" else "LOG_HOURS_PER_NM"
        return np.expm1(df[log_col].to_numpy(dtype=float))
    return df[spec.target_column].to_numpy(dtype=float)


def compute_permutation_importance(
    payload: dict,
    metadata: dict,
    spec: DatasetSpec,
    df: pd.DataFrame,
    repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, float]:
    feature_columns = list(payload["feature_columns"])
    X = prepare_features(df, metadata, feature_columns)
    y = target_in_original_units(df, payload, spec)

    baseline_pred = predict_in_original_units(payload, X)
    baseline_mae = float(np.mean(np.abs(baseline_pred - y)))

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | str]] = []
    for feature in feature_columns:
        deltas: list[float] = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy(copy=True))
            perm_pred = predict_in_original_units(payload, shuffled)
            perm_mae = float(np.mean(np.abs(perm_pred - y)))
            deltas.append(perm_mae - baseline_mae)
        rows.append(
            {
                "feature": feature,
                "baseline_mae": baseline_mae,
                "importance_mean": float(np.mean(deltas)),
                "importance_std": float(np.std(deltas, ddof=0)),
                "importance_min": float(np.min(deltas)),
                "importance_max": float(np.max(deltas)),
            }
        )

    importance = pd.DataFrame(rows).sort_values(
        ["importance_mean", "importance_std"], ascending=[False, False]
    )
    return importance.reset_index(drop=True), baseline_mae


def save_plot(importance: pd.DataFrame, baseline_mae: float, out_path: Path, title: str, top_k: int) -> None:
    top = importance.head(top_k).iloc[::-1]
    fig_height = max(4.0, 0.38 * len(top) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.barh(top["feature"], top["importance_mean"], color="#555555")
    ax.set_xlabel("Validation MAE increase after permutation")
    ax.set_ylabel("Feature")
    ax.set_title(f"{title}\nBaseline MAE = {baseline_mae:.4f}")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_summary(
    spec: DatasetSpec,
    split: str,
    artifact_filename: str,
    baseline_mae: float,
    importance: pd.DataFrame,
    out_path: Path,
) -> None:
    top = importance.head(10)
    lines = [
        f"Permutation importance summary: {spec.split_report_label}",
        f"Split: {split}",
        f"Artifact: Models/Artifacts/<customer>/{artifact_filename}",
        f"Baseline MAE: {baseline_mae:.6f}",
        "",
        "Top features by validation MAE increase:",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"- {row['feature']}: mean_delta={row['importance_mean']:.6f}, std={row['importance_std']:.6f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(paths: CustomerPaths, spec: DatasetSpec, args: argparse.Namespace) -> None:
    metadata = load_metadata(paths, spec)
    payload = load_artifact(paths, spec)
    df = load_split(metadata, args.split)

    importance, baseline_mae = compute_permutation_importance(
        payload=payload,
        metadata=metadata,
        spec=spec,
        df=df,
        repeats=args.repeats,
        random_state=args.random_state,
    )

    qa_dir = paths.derived_dir / QA_ML_SUBDIR
    csv_path = qa_dir / f"{spec.output_prefix}_{args.split}_permutation_importance.csv"
    png_path = qa_dir / f"{spec.output_prefix}_{args.split}_permutation_importance.png"
    txt_path = qa_dir / f"{spec.output_prefix}_{args.split}_permutation_importance_summary.txt"

    importance.to_csv(csv_path, index=False)
    save_plot(
        importance=importance,
        baseline_mae=baseline_mae,
        out_path=png_path,
        title=f"{paths.key} {spec.split_report_label.title()} permutation importance",
        top_k=args.top_k,
    )
    write_summary(
        spec=spec,
        split=args.split,
        artifact_filename=spec.artifact_filename,
        baseline_mae=baseline_mae,
        importance=importance,
        out_path=txt_path,
    )

    print(f"[{paths.key}] {spec.name} diagnostics saved:")
    print(f"  - {csv_path.relative_to(ROOT)}")
    print(f"  - {png_path.relative_to(ROOT)}")
    print(f"  - {txt_path.relative_to(ROOT)}")


def main() -> None:
    args = parse_args()
    if args.list_customers:
        print(describe_customers())
        return

    paths = configure_customer(args.customer)
    for dataset_name in args.dataset:
        run_one(paths, DATASETS[dataset_name], args)


if __name__ == "__main__":
    main()
