from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

from Models.fit_sailing_time_model import build_pipeline, compute_metrics  # noqa: E402
from Models.optimizer import ModelAdapters  # noqa: E402
from Models.optimizer_adapters import SailingTimeModelAdapter, TurnaroundModelAdapter  # noqa: E402
from Models.optimizer_mip import solve_mip  # noqa: E402
from Models.run_thesis_optimizer_demo import (  # noqa: E402
    DemoConfig,
    add_haversine_miles,
    build_request,
    extract_reposition_diagnostics,
    load_sample,
    prep_dates,
    write_outputs,
)
from Models.sweep_ml_hyperparams import lookup_metrics  # noqa: E402
from Models.train_sailing_time_model import (  # noqa: E402
    LOG_TARGET_COLUMN,
    TARGET_COLUMN,
    configure_customer,
    enrich_features,
    load_training,
    perform_ratio_split,
)

SAFE_VARIANT = "planning_safe"
VALIDATION_RATIO = 0.30
SWEEP_RANDOM_STATE = 42

SAFE_CATEGORICAL_FEATURES = [
    "VESSEL_TYPE_ID",
    "HAS_CANAL_PASSAGE",
    "MONTH_NO",
    "SEASON",
]
SAFE_DERIVED_CATEGORICAL_FEATURES = [
    "VESSEL_TYPE_CANAL_KEY",
    "VESSEL_TYPE_MONTH_KEY",
]
SAFE_NUMERIC_FEATURES = [
    "BALLAST_FRAC",
    "MILES_TOTAL",
    "MILES_BALLAST",
    "MILES_LOADED",
    "DWT_SUMMER",
    "DRAFT_SUMMER",
    "LOA",
    "BEAM",
    "TOTAL_CARGO_QUANTITY",
    "N_CARGO_ROWS",
    "N_UNIQUE_CARGO",
    "VESSEL_AGE_YEARS",
    "CANAL_COST",
]

LOOKUP_HIERARCHY = (
    ("VESSEL_TYPE_ID", "HAS_CANAL_PASSAGE", "MONTH_NO"),
    ("VESSEL_TYPE_ID", "HAS_CANAL_PASSAGE"),
    ("VESSEL_TYPE_ID",),
    (),
)


def compute_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def prepare_safe_dataset(customer: str = "customer1") -> tuple[pd.DataFrame, pd.DataFrame, dict, Any]:
    paths = configure_customer(customer)
    raw_training = load_training(paths)
    enriched = enrich_features(raw_training)

    selected_columns = (
        SAFE_CATEGORICAL_FEATURES
        + SAFE_DERIVED_CATEGORICAL_FEATURES
        + SAFE_NUMERIC_FEATURES
        + ["VOYAGE_ID", "VOYAGE_START_DATE", TARGET_COLUMN, LOG_TARGET_COLUMN]
    )
    missing = [col for col in selected_columns if col not in enriched.columns]
    if missing:
        raise ValueError(f"Expected planning-safe columns missing from sailing dataset: {', '.join(missing)}")

    dataset = enriched[selected_columns].copy()
    train_df, val_df, cutoff_ts = perform_ratio_split(dataset, VALIDATION_RATIO)

    ml_dir = paths.derived_dir / "ML"
    train_path = ml_dir / f"sailing_time_train_{SAFE_VARIANT}.parquet"
    val_path = ml_dir / f"sailing_time_validation_{SAFE_VARIANT}.parquet"
    metadata_path = ml_dir / f"sailing_time_features_{SAFE_VARIANT}.json"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    metadata = {
        "customer": paths.key,
        "feature_set": SAFE_VARIANT,
        "planning_time_safe": True,
        "split_strategy": {
            "type": "ratio",
            "validation_ratio": VALIDATION_RATIO,
            "n_train": int(len(train_df)),
            "n_validation": int(len(val_df)),
        },
        "cutoff_date": cutoff_ts.strftime("%Y-%m-%d") if cutoff_ts else None,
        "target_column": TARGET_COLUMN,
        "log_target_column": LOG_TARGET_COLUMN,
        "categorical_features": SAFE_CATEGORICAL_FEATURES,
        "derived_categorical_features": SAFE_DERIVED_CATEGORICAL_FEATURES,
        "numeric_features": SAFE_NUMERIC_FEATURES,
        "additional_features": [],
        "excluded_for_planning_safe": [
            "DAYS_AT_SEA",
            "DAYS_IN_PORT_TOTAL",
            "DAYS_TOTAL",
            "VOYAGE_START_TS",
            "VOYAGE_YEAR_MONTH",
        ],
        "train_path": str(train_path.relative_to(ROOT)),
        "validation_path": str(val_path.relative_to(ROOT)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return train_df, val_df, metadata, paths


def build_args(
    *,
    model: str,
    max_depth: int,
    min_samples_leaf: int,
    use_log_target: bool,
    learning_rate: float = 0.05,
    max_leaf_nodes: int = 31,
    max_iter: int = 300,
    label_suffix: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=SWEEP_RANDOM_STATE,
        use_log_target=use_log_target,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        max_iter=max_iter,
        label_suffix=label_suffix,
    )


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    metadata: dict,
    *,
    model: str,
    max_depth: int,
    min_samples_leaf: int,
    use_log_target: bool,
    learning_rate: float = 0.05,
    max_leaf_nodes: int = 31,
    max_iter: int = 300,
) -> tuple[dict[str, Any], Pipeline, np.ndarray, np.ndarray]:
    args = build_args(
        model=model,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        use_log_target=use_log_target,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        max_iter=max_iter,
    )
    categorical = metadata["categorical_features"] + metadata["derived_categorical_features"]
    numeric = metadata["numeric_features"]
    feature_columns = categorical + numeric

    X_train = train_df[feature_columns].copy()
    X_val = val_df[feature_columns].copy()

    for col in categorical:
        X_train[col] = X_train[col].astype("string").fillna("MISSING")
        X_val[col] = X_val[col].astype("string").fillna("MISSING")
    for col in numeric:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_val[col] = pd.to_numeric(X_val[col], errors="coerce")

    target_col = LOG_TARGET_COLUMN if use_log_target else TARGET_COLUMN
    y_train = train_df[target_col].to_numpy()
    y_val = val_df[target_col].to_numpy()

    pipeline = build_pipeline(model, categorical, numeric, args)
    pipeline.fit(X_train, y_train)

    train_pred = np.asarray(pipeline.predict(X_train), dtype=float)
    val_pred = np.asarray(pipeline.predict(X_val), dtype=float)

    if use_log_target:
        train_true_eval = np.expm1(y_train)
        val_true_eval = np.expm1(y_val)
        train_pred_eval = np.expm1(train_pred)
        val_pred_eval = np.expm1(val_pred)
    else:
        train_true_eval = train_df[TARGET_COLUMN].to_numpy(dtype=float)
        val_true_eval = val_df[TARGET_COLUMN].to_numpy(dtype=float)
        train_pred_eval = train_pred
        val_pred_eval = val_pred

    metrics = {
        "train": compute_metrics(train_true_eval, train_pred_eval),
        "validation": compute_metrics(val_true_eval, val_pred_eval),
        "bias": {
            "train": compute_bias(train_true_eval, train_pred_eval),
            "validation": compute_bias(val_true_eval, val_pred_eval),
        },
    }
    return metrics, pipeline, train_pred_eval, val_pred_eval


def save_artifact(paths, metadata: dict, pipeline, metrics: dict, *, suffix: str, use_log_target: bool, params: dict[str, Any]) -> tuple[Path, Path]:
    artifact_path = ROOT / "Models" / "Artifacts" / paths.key / f"sailing_time_{suffix}.joblib"
    metrics_path = ROOT / "Models" / "Artifacts" / paths.key / f"sailing_time_{suffix}_metrics.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "target_column": TARGET_COLUMN,
        "use_log_target": use_log_target,
        "feature_columns": metadata["categorical_features"] + metadata["derived_categorical_features"] + metadata["numeric_features"],
        "metadata": metadata,
        "model_type": params["model"],
        "label_suffix": SAFE_VARIANT,
    }
    joblib.dump(payload, artifact_path)
    metrics_payload = {"metrics": metrics, "parameters": params}
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return artifact_path, metrics_path


def run_hgbt_sweep(train_df: pd.DataFrame, val_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for max_depth in (6, 8):
        for min_samples_leaf in (10, 20):
            for learning_rate in (0.03, 0.05):
                for max_leaf_nodes in (31, 63):
                    for use_log_target in (False, True):
                        metrics, _, _, _ = train_model(
                            train_df,
                            val_df,
                            metadata,
                            model="hgbt",
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            use_log_target=use_log_target,
                            learning_rate=learning_rate,
                            max_leaf_nodes=max_leaf_nodes,
                            max_iter=300,
                        )
                        rows.append(
                            {
                                "model_type": "hgbt",
                                "max_depth": max_depth,
                                "min_samples_leaf": min_samples_leaf,
                                "learning_rate": learning_rate,
                                "max_leaf_nodes": max_leaf_nodes,
                                "max_iter": 300,
                                "use_log_target": use_log_target,
                                "train_mae": metrics["train"]["mae"],
                                "train_rmse": metrics["train"]["rmse"],
                                "train_smape": metrics["train"]["smape"],
                                "val_mae": metrics["validation"]["mae"],
                                "val_rmse": metrics["validation"]["rmse"],
                                "val_smape": metrics["validation"]["smape"],
                                "val_bias": metrics["bias"]["validation"],
                            }
                        )
    return pd.DataFrame(rows).sort_values(["val_mae", "val_rmse", "train_mae"]).reset_index(drop=True)


def compute_permutation_importance(payload: dict, metadata: dict, val_df: pd.DataFrame, repeats: int = 20) -> pd.DataFrame:
    feature_columns = payload["feature_columns"]
    X = val_df[feature_columns].copy()
    categorical = metadata["categorical_features"] + metadata["derived_categorical_features"]
    numeric = metadata["numeric_features"]
    for col in categorical:
        X[col] = X[col].astype("string").fillna("MISSING")
    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    y = val_df[TARGET_COLUMN].to_numpy(dtype=float)
    baseline_raw = np.asarray(payload["pipeline"].predict(X), dtype=float)
    baseline_pred = np.expm1(baseline_raw) if payload.get("use_log_target", False) else baseline_raw
    baseline_mae = float(np.mean(np.abs(baseline_pred - y)))

    rng = np.random.default_rng(SWEEP_RANDOM_STATE)
    rows = []
    for feature in feature_columns:
        deltas = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy(copy=True))
            pred_raw = np.asarray(payload["pipeline"].predict(shuffled), dtype=float)
            pred = np.expm1(pred_raw) if payload.get("use_log_target", False) else pred_raw
            deltas.append(float(np.mean(np.abs(pred - y))) - baseline_mae)
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
    return pd.DataFrame(rows).sort_values(["importance_mean", "importance_std"], ascending=[False, False]).reset_index(drop=True)


def save_importance_outputs(paths, importance: pd.DataFrame) -> tuple[Path, Path, Path]:
    qa_dir = paths.derived_dir / "QA" / "ml"
    csv_path = qa_dir / f"sailing_time_{SAFE_VARIANT}_hgbt_permutation_importance.csv"
    txt_path = qa_dir / f"sailing_time_{SAFE_VARIANT}_hgbt_permutation_importance_summary.txt"
    png_path = qa_dir / f"sailing_time_{SAFE_VARIANT}_hgbt_permutation_importance.png"
    importance.to_csv(csv_path, index=False)

    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.38 * len(top) + 1.5)))
    ax.barh(top["feature"], top["importance_mean"], color="#555555")
    ax.set_xlabel("Validation MAE increase after permutation")
    ax.set_ylabel("Feature")
    baseline_mae = float(importance["baseline_mae"].iloc[0]) if not importance.empty else float("nan")
    ax.set_title(f"Planning-safe sailing HGBT permutation importance\nBaseline MAE = {baseline_mae:.4f}")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    lines = [
        "Permutation importance summary: sailing time (planning-safe HGBT)",
        f"Baseline MAE: {baseline_mae:.6f}",
        "",
        "Top features by validation MAE increase:",
    ]
    for _, row in importance.head(10).iterrows():
        lines.append(
            f"- {row['feature']}: mean_delta={row['importance_mean']:.6f}, std={row['importance_std']:.6f}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, txt_path, png_path


def rerun_optimizer_baseline(sailing_artifact: Path, sailing_metadata: dict) -> dict[str, Any]:
    turnaround_meta = json.loads(
        (ROOT / "DataSets" / "Derived" / "Customer1" / "ML" / "port_turnaround_features.json").read_text()
    )
    turnaround_artifact = ROOT / "Models" / "Artifacts" / "customer1" / "port_turnaround_dt.joblib"

    adapters = ModelAdapters(
        turnaround=TurnaroundModelAdapter(turnaround_artifact, turnaround_meta),
        sailing_time=SailingTimeModelAdapter(sailing_artifact, sailing_metadata),
    )

    vessels, voyages, fleet_plan, manual = load_sample()
    prep_dates(vessels, voyages)
    add_haversine_miles(voyages)
    request = build_request(vessels, voyages, fleet_plan, manual)
    pref = (
        dict(zip(manual["VOYAGE_ID"].astype(str), manual["VESSEL_ID"].astype(str)))
        if manual is not None and not manual.empty
        else None
    )

    cfg = DemoConfig(time_limit_sec=60)
    result = solve_mip(
        request,
        adapters,
        max_arcs_per_job=cfg.max_arcs_per_job,
        time_limit_sec=cfg.time_limit_sec,
        window_slack_days=3.0,
        unserved_penalty=cfg.unserved_penalty_physical,
        hard_laycan_end=True,
        job_job_selection=cfg.job_job_selection,
        job_job_candidate_window=cfg.job_job_candidate_window,
        preferred_vessels=pref,
        switch_penalty=0.5,
        vessel_used_penalty=0.0,
        missing_reposition_days=1.0,
    )
    out = write_outputs(
        prefix="thesis_aligned116_sailing_plan_safe",
        tag="manual_match_physical",
        result=result,
        manual=manual,
        out_dir=ROOT / "Visualizations" / "output",
        overlap_tolerance_sec=60.0,
    )
    return {
        "result": result,
        "sched_stats": out["sched_stats"],
        "manual_stats": out["manual_stats"],
        "reposition_diag": out["reposition_diag"],
        "eval_path": out["eval"],
        "allocations_path": out["allocations"],
        "schedule_path": out["schedule"],
    }


def write_summary(
    paths,
    metadata: dict,
    lookup: dict[str, float],
    locked_safe_metrics: dict[str, Any],
    best_hgbt_params: dict[str, Any],
    best_hgbt_metrics: dict[str, Any],
    artifact_dt: Path,
    artifact_hgbt: Path,
    importance_paths: tuple[Path, Path, Path],
    optimizer_result: dict[str, Any],
) -> Path:
    txt_path = ROOT / "TextFiles" / "sailing_time_planning_safe_comparison.txt"
    sched = optimizer_result["sched_stats"]
    result = optimizer_result["result"]
    lines = [
        "Planning-time-safe sailing-time model comparison",
        "Date: 2026-05-22",
        "",
        "Purpose",
        "- Re-evaluate the sailing-time model using a planning-time-safe feature set that excludes realised sailing duration and similar post-voyage information.",
        "",
        "1) Exact feature list used in the planning-safe run",
        "Categorical features:",
        *[f"- {feat}" for feat in metadata["categorical_features"]],
        "Derived categorical features:",
        *[f"- {feat}" for feat in metadata["derived_categorical_features"]],
        "Numeric features:",
        *[f"- {feat}" for feat in metadata["numeric_features"]],
        "Explicitly excluded from the planning-safe feature set:",
        *[f"- {feat}" for feat in metadata["excluded_for_planning_safe"]],
        "",
        "2) Train/validation split sizes",
        f"- train rows: {metadata['split_strategy']['n_train']}",
        f"- validation rows: {metadata['split_strategy']['n_validation']}",
        f"- split strategy: chronological ratio with validation_ratio = {metadata['split_strategy']['validation_ratio']}",
        "",
        "3) Lookup baseline on the same planning-safe validation split",
        f"- MAE: {lookup['mae']}",
        f"- RMSE: {lookup['rmse']}",
        f"- mean residual: {lookup['bias']}",
        "",
        "4) Locked decision-tree specification rerun on the same planning-safe feature set",
        f"- artifact: {artifact_dt.relative_to(ROOT)}",
        f"- MAE: {locked_safe_metrics['validation']['mae']}",
        f"- RMSE: {locked_safe_metrics['validation']['rmse']}",
        f"- mean residual: {locked_safe_metrics['bias']['validation']}",
        "",
        "5) Best tuned HGBT on the planning-safe feature set",
        f"- artifact: {artifact_hgbt.relative_to(ROOT)}",
        f"- MAE: {best_hgbt_metrics['validation']['mae']}",
        f"- RMSE: {best_hgbt_metrics['validation']['rmse']}",
        f"- mean residual: {best_hgbt_metrics['bias']['validation']}",
        "",
        "6) Best HGBT hyperparameters",
        f"- max_depth: {best_hgbt_params['max_depth']}",
        f"- min_samples_leaf: {best_hgbt_params['min_samples_leaf']}",
        f"- learning_rate: {best_hgbt_params['learning_rate']}",
        f"- max_leaf_nodes: {best_hgbt_params['max_leaf_nodes']}",
        f"- max_iter: {best_hgbt_params['max_iter']}",
        f"- use_log_target: {best_hgbt_params['use_log_target']}",
        "",
        "7) Permutation feature importance for the best planning-safe HGBT model",
        f"- csv: {importance_paths[0].relative_to(ROOT)}",
        f"- summary: {importance_paths[1].relative_to(ROOT)}",
        f"- plot: {importance_paths[2].relative_to(ROOT)}",
        "",
        "8) Does the tuned HGBT still beat the lookup baseline?",
        (
            f"- yes: HGBT validation MAE {best_hgbt_metrics['validation']['mae']} < lookup MAE {lookup['mae']}"
            if best_hgbt_metrics["validation"]["mae"] < lookup["mae"]
            else f"- no: HGBT validation MAE {best_hgbt_metrics['validation']['mae']} >= lookup MAE {lookup['mae']}"
        ),
        "",
        "9) Saved artifact names and paths",
        f"- metadata: DataSets/Derived/Customer1/ML/sailing_time_features_{SAFE_VARIANT}.json",
        f"- train split: DataSets/Derived/Customer1/ML/sailing_time_train_{SAFE_VARIANT}.parquet",
        f"- validation split: DataSets/Derived/Customer1/ML/sailing_time_validation_{SAFE_VARIANT}.parquet",
        f"- DT artifact: {artifact_dt.relative_to(ROOT)}",
        f"- HGBT artifact: {artifact_hgbt.relative_to(ROOT)}",
        "",
        "10) Optimiser baseline rerun with the planning-safe sailing HGBT model",
        "- yes",
        f"- served voyages: {sched['n_actions']}",
        f"- unserved voyages: {116 - sched['n_actions']}",
        f"- laycan start violations: {sched['laycan_start_violations']}",
        f"- laycan end violations: {sched['laycan_end_violations']}",
        f"- overlaps: {sched['overlaps']}",
        f"- objective value: {getattr(result, 'objective', None)}",
        f"- eval file: {Path(optimizer_result['eval_path']).relative_to(ROOT)}",
        "",
        "Interpretation",
        "- This comparison is stricter than the previous sailing-time tuning result because it removes `DAYS_AT_SEA` and related realised-duration information.",
        "- The result should therefore be treated as the more planning-relevant sailing-time comparison.",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return txt_path


def main() -> None:
    train_df, val_df, metadata, paths = prepare_safe_dataset()

    # Lookup baseline on same validation split
    lookup = lookup_metrics(
        paths,
        type("Spec", (), {
            "name": "sailing",
            "lookup_filename": "sailing_time_lookup.csv",
            "hierarchy": LOOKUP_HIERARCHY,
            "lookup_value_column": "median_hours_per_nm",
            "target_column": TARGET_COLUMN,
        })(),
        val_df,
    )

    # Locked DT specification rerun on planning-safe set
    dt_metrics, dt_pipeline, _, _ = train_model(
        train_df,
        val_df,
        metadata,
        model="tree",
        max_depth=8,
        min_samples_leaf=20,
        use_log_target=False,
    )
    dt_params = {
        "model": "tree",
        "max_depth": 8,
        "min_samples_leaf": 20,
        "random_state": SWEEP_RANDOM_STATE,
        "use_log_target": False,
        "label_suffix": SAFE_VARIANT,
    }
    artifact_dt, _ = save_artifact(
        paths,
        metadata,
        dt_pipeline,
        dt_metrics,
        suffix=f"dt_{SAFE_VARIANT}",
        use_log_target=False,
        params=dt_params,
    )

    # Tuned HGBT sweep on planning-safe set
    sweep_df = run_hgbt_sweep(train_df, val_df, metadata)
    qa_dir = paths.derived_dir / "QA" / "ml"
    sweep_csv = qa_dir / f"sailing_time_{SAFE_VARIANT}_hyperparam_sweep.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    best = sweep_df.iloc[0].to_dict()
    hgbt_metrics, hgbt_pipeline, _, _ = train_model(
        train_df,
        val_df,
        metadata,
        model="hgbt",
        max_depth=int(best["max_depth"]),
        min_samples_leaf=int(best["min_samples_leaf"]),
        use_log_target=bool(best["use_log_target"]),
        learning_rate=float(best["learning_rate"]),
        max_leaf_nodes=int(best["max_leaf_nodes"]),
        max_iter=int(best["max_iter"]),
    )
    hgbt_params = {
        "model": "hgbt",
        "max_depth": int(best["max_depth"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "random_state": SWEEP_RANDOM_STATE,
        "use_log_target": bool(best["use_log_target"]),
        "learning_rate": float(best["learning_rate"]),
        "max_leaf_nodes": int(best["max_leaf_nodes"]),
        "max_iter": int(best["max_iter"]),
        "label_suffix": SAFE_VARIANT,
    }
    artifact_hgbt, _ = save_artifact(
        paths,
        metadata,
        hgbt_pipeline,
        hgbt_metrics,
        suffix=f"hgbt_{SAFE_VARIANT}",
        use_log_target=bool(best["use_log_target"]),
        params=hgbt_params,
    )

    payload = joblib.load(artifact_hgbt)
    importance = compute_permutation_importance(payload, metadata, val_df)
    importance_paths = save_importance_outputs(paths, importance)

    optimizer_result = rerun_optimizer_baseline(artifact_hgbt, metadata)
    summary_path = write_summary(
        paths,
        metadata,
        lookup,
        dt_metrics,
        best,
        hgbt_metrics,
        artifact_dt,
        artifact_hgbt,
        importance_paths,
        optimizer_result,
    )

    print(f"Saved planning-safe sailing comparison summary -> {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
