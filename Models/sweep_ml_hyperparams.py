from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

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
QA_ML_SUBDIR = Path("QA") / "ml"
ARTIFACT_DIR = ROOT / "Models" / "Artifacts"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    metadata_filename: str
    target_column: str
    current_model_artifact: str
    current_model_metrics: str
    lookup_filename: str
    output_prefix: str
    hierarchy: tuple[tuple[str, ...], ...]
    lookup_value_column: str


DATASETS: dict[str, DatasetSpec] = {
    "turnaround": DatasetSpec(
        name="turnaround",
        metadata_filename="port_turnaround_features.json",
        target_column="DAYS_IN_PORT",
        current_model_artifact="port_turnaround_dt.joblib",
        current_model_metrics="port_turnaround_dt_metrics.json",
        lookup_filename="port_turnaround_lookup.csv",
        output_prefix="port_turnaround",
        hierarchy=(
            ("PORT_ID", "TERMINAL_ID", "IS_BALLAST", "VESSEL_TYPE_ID", "MONTH_NO"),
            ("PORT_ID", "TERMINAL_ID", "IS_BALLAST", "VESSEL_TYPE_ID"),
            ("PORT_ID", "TERMINAL_ID", "IS_BALLAST"),
            ("PORT_ID", "IS_BALLAST"),
            ("PORT_ID",),
            (),
        ),
        lookup_value_column="median_days_in_port",
    ),
    "sailing": DatasetSpec(
        name="sailing",
        metadata_filename="sailing_time_features.json",
        target_column="HOURS_PER_NM",
        current_model_artifact="sailing_time_dt.joblib",
        current_model_metrics="sailing_time_dt_metrics.json",
        lookup_filename="sailing_time_lookup.csv",
        output_prefix="sailing_time",
        hierarchy=(
            ("VESSEL_TYPE_ID", "HAS_CANAL_PASSAGE", "MONTH_NO"),
            ("VESSEL_TYPE_ID", "HAS_CANAL_PASSAGE"),
            ("VESSEL_TYPE_ID",),
            (),
        ),
        lookup_value_column="median_hours_per_nm",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded chronological hyperparameter sweep for thesis ML models."
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
        help="Which dataset sweeps to run.",
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


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected split file missing: {path}")
    return pd.read_parquet(path)


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    epsilon = 1e-8
    denom = np.abs(y_true) + np.abs(y_pred) + epsilon
    smape = 2.0 * np.abs(y_true - y_pred) / denom
    return float(np.mean(smape) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    smape = compute_smape(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "smape": smape,
        "bias": bias,
    }


def build_pipeline(
    model_type: str,
    categorical_features: list[str],
    numeric_features: list[str],
    params: dict[str, Any],
) -> Pipeline:
    encoder_kwargs = {"handle_unknown": "ignore", "sparse_output": False}
    try:
        OneHotEncoder(**encoder_kwargs)
    except TypeError:
        encoder_kwargs = {"handle_unknown": "ignore", "sparse": False}

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(**encoder_kwargs)),
                    ]
                ),
                categorical_features,
            ),
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
        ]
    )

    if model_type == "tree":
        model = DecisionTreeRegressor(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=42,
        )
    else:
        model = HistGradientBoostingRegressor(
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            max_leaf_nodes=params["max_leaf_nodes"],
            max_iter=params["max_iter"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=42,
        )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def prepare_features(df: pd.DataFrame, categorical: list[str], numeric: list[str]) -> pd.DataFrame:
    X = df[categorical + numeric].copy()
    for col in categorical:
        if col in X.columns:
            X[col] = X[col].astype("string").fillna("MISSING")
    for col in numeric:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def evaluate_config(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    spec: DatasetSpec,
    model_type: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    feature_columns = categorical + numeric
    target_col = spec.target_column
    if params.get("use_log_target"):
        log_col = "LOG_DAYS_IN_PORT" if spec.name == "turnaround" else "LOG_HOURS_PER_NM"
        if log_col not in train_df.columns:
            raise ValueError(f"Expected log target column missing: {log_col}")
        target_col = log_col

    X_train = prepare_features(train_df, categorical, numeric)
    X_val = prepare_features(val_df, categorical, numeric)
    y_train = train_df[target_col].to_numpy()
    y_val = val_df[target_col].to_numpy()

    pipeline = build_pipeline(model_type, categorical, numeric, params)
    pipeline.fit(X_train, y_train)

    train_pred = np.asarray(pipeline.predict(X_train), dtype=float)
    val_pred = np.asarray(pipeline.predict(X_val), dtype=float)

    if params.get("use_log_target"):
        train_true_eval = np.expm1(y_train)
        val_true_eval = np.expm1(y_val)
        train_pred_eval = np.expm1(train_pred)
        val_pred_eval = np.expm1(val_pred)
    else:
        train_true_eval = train_df[spec.target_column].to_numpy(dtype=float)
        val_true_eval = val_df[spec.target_column].to_numpy(dtype=float)
        train_pred_eval = train_pred
        val_pred_eval = val_pred

    train_metrics = compute_metrics(train_true_eval, train_pred_eval)
    val_metrics = compute_metrics(val_true_eval, val_pred_eval)

    result = {
        "model_type": model_type,
        **params,
        "train_mae": train_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "train_smape": train_metrics["smape"],
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_smape": val_metrics["smape"],
        "val_bias": val_metrics["bias"],
    }
    return result


def norm_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def build_lookup_maps(lookup_df: pd.DataFrame, hierarchy: tuple[tuple[str, ...], ...], value_column: str) -> dict[tuple[str, ...], dict[tuple[Any, ...], float]]:
    maps: dict[tuple[str, ...], dict[tuple[Any, ...], float]] = {}
    for level_index, keys in enumerate(hierarchy, start=1):
        if "level" in lookup_df.columns:
            level_df = lookup_df.loc[lookup_df["level"] == level_index].copy()
        else:
            level_df = lookup_df.copy()
        if not keys:
            value = float(level_df[value_column].dropna().iloc[0])
            maps[keys] = {(): value}
            continue

        subset = level_df[list(keys) + [value_column]].copy()
        subset = subset.dropna(subset=[value_column])
        key_map: dict[tuple[Any, ...], float] = {}
        for _, row in subset.iterrows():
            key = tuple(norm_value(row[col]) for col in keys)
            key_map[key] = float(row[value_column])
        maps[keys] = key_map
    return maps


def predict_lookup(
    df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    hierarchy: tuple[tuple[str, ...], ...],
    value_column: str,
) -> np.ndarray:
    lookup_maps = build_lookup_maps(lookup_df, hierarchy, value_column)
    preds: list[float] = []
    for _, row in df.iterrows():
        predicted = None
        for keys in hierarchy:
            if not keys:
                predicted = lookup_maps[keys][()]
                break
            key = tuple(norm_value(row.get(col)) for col in keys)
            if key in lookup_maps[keys]:
                predicted = lookup_maps[keys][key]
                break
        preds.append(float(predicted))
    return np.asarray(preds, dtype=float)


def lookup_metrics(paths: CustomerPaths, spec: DatasetSpec, val_df: pd.DataFrame) -> dict[str, float]:
    if spec.name == "turnaround":
        lookup_path = paths.derived_dir / spec.lookup_filename
    else:
        lookup_path = paths.derived_dir / "sailing_time" / spec.lookup_filename
    lookup_df = pd.read_csv(lookup_path)
    pred = predict_lookup(val_df, lookup_df, spec.hierarchy, spec.lookup_value_column)
    actual = val_df[spec.target_column].to_numpy(dtype=float)
    return compute_metrics(actual, pred)


def current_model_metrics(paths: CustomerPaths, spec: DatasetSpec) -> dict[str, Any]:
    metrics_path = ARTIFACT_DIR / paths.key / spec.current_model_metrics
    payload = json.loads(metrics_path.read_text())
    return payload["metrics"]["validation"]


def search_space() -> list[tuple[str, dict[str, Any]]]:
    configs: list[tuple[str, dict[str, Any]]] = []

    tree_grid = {
        "max_depth": [6, 8, 12],
        "min_samples_leaf": [10, 20, 40],
        "use_log_target": [False],
    }
    for values in itertools.product(*tree_grid.values()):
        configs.append(("tree", dict(zip(tree_grid.keys(), values))))

    hgbt_grid = {
        "max_depth": [6, 8],
        "min_samples_leaf": [10, 20],
        "learning_rate": [0.03, 0.05],
        "max_leaf_nodes": [31, 63],
        "max_iter": [300],
        "use_log_target": [False, True],
    }
    for values in itertools.product(*hgbt_grid.values()):
        configs.append(("hgbt", dict(zip(hgbt_grid.keys(), values))))

    return configs


def write_summary(
    spec: DatasetSpec,
    metadata: dict,
    current_metrics: dict[str, Any],
    lookup: dict[str, float],
    results: pd.DataFrame,
    out_path: Path,
) -> None:
    best = results.iloc[0]
    lines = [
        f"Bounded hyperparameter sweep summary: {spec.name}",
        f"Split strategy: {metadata.get('split_strategy', {})}",
        f"Rows searched: {len(results)}",
        "",
        "Current locked thesis model validation:",
        f"- artifact: Models/Artifacts/<customer>/{spec.current_model_artifact}",
        f"- MAE: {current_metrics['mae']:.6f}",
        f"- RMSE: {current_metrics['rmse']:.6f}",
        "",
        "Lookup baseline validation:",
        f"- MAE: {lookup['mae']:.6f}",
        f"- RMSE: {lookup['rmse']:.6f}",
        "",
        "Best ML configuration from bounded sweep:",
        f"- model_type: {best['model_type']}",
        f"- max_depth: {best['max_depth']}",
        f"- min_samples_leaf: {best['min_samples_leaf']}",
        f"- learning_rate: {best['learning_rate']}",
        f"- max_leaf_nodes: {best['max_leaf_nodes']}",
        f"- max_iter: {best['max_iter']}",
        f"- use_log_target: {best['use_log_target']}",
        f"- validation MAE: {best['val_mae']:.6f}",
        f"- validation RMSE: {best['val_rmse']:.6f}",
        "",
        "Best-vs-current delta:",
        f"- delta MAE: {best['val_mae'] - current_metrics['mae']:.6f}",
        f"- delta RMSE: {best['val_rmse'] - current_metrics['rmse']:.6f}",
        "",
        "Best-vs-lookup delta:",
        f"- delta MAE: {best['val_mae'] - lookup['mae']:.6f}",
        f"- delta RMSE: {best['val_rmse'] - lookup['rmse']:.6f}",
        "",
        "Top 5 configurations by validation MAE:",
    ]
    for _, row in results.head(5).iterrows():
        lines.append(
            "- "
            + ", ".join(
                [
                    f"model={row['model_type']}",
                    f"depth={row['max_depth']}",
                    f"leaf={row['min_samples_leaf']}",
                    f"lr={row['learning_rate']}",
                    f"nodes={row['max_leaf_nodes']}",
                    f"iter={row['max_iter']}",
                    f"log={row['use_log_target']}",
                    f"val_mae={row['val_mae']:.6f}",
                    f"val_rmse={row['val_rmse']:.6f}",
                ]
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(paths: CustomerPaths, spec: DatasetSpec) -> None:
    metadata = load_metadata(paths, spec)
    train_df = load_split(ROOT / metadata["train_path"])
    val_df = load_split(ROOT / metadata["validation_path"])

    categorical = metadata.get("categorical_features", []) + metadata.get(
        "derived_categorical_features", []
    )
    numeric = metadata.get("numeric_features", []) + metadata.get("derived_numeric_features", [])

    results: list[dict[str, Any]] = []
    for model_type, params in search_space():
        record = evaluate_config(train_df, val_df, categorical, numeric, spec, model_type, params)
        results.append(record)

    results_df = pd.DataFrame(results).sort_values(
        ["val_mae", "val_rmse", "train_mae"], ascending=[True, True, True]
    ).reset_index(drop=True)

    lookup = lookup_metrics(paths, spec, val_df)
    current = current_model_metrics(paths, spec)

    qa_dir = paths.derived_dir / QA_ML_SUBDIR
    csv_path = qa_dir / f"{spec.output_prefix}_hyperparam_sweep.csv"
    txt_path = qa_dir / f"{spec.output_prefix}_hyperparam_sweep_summary.txt"

    results_df.to_csv(csv_path, index=False)
    write_summary(spec, metadata, current, lookup, results_df, txt_path)

    print(f"[{paths.key}] {spec.name} sweep saved:")
    print(f"  - {csv_path.relative_to(ROOT)}")
    print(f"  - {txt_path.relative_to(ROOT)}")


def main() -> None:
    args = parse_args()
    if args.list_customers:
        print(describe_customers())
        return

    paths = configure_customer(args.customer)
    for dataset_name in args.dataset:
        run_one(paths, DATASETS[dataset_name])


if __name__ == "__main__":
    main()
