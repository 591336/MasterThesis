from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

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
DERIVED_SUBDIR = "sailing_time"
TRAINING_FILENAME = "sailing_time_training.csv"

TARGET_COLUMN = "HOURS_PER_NM"
LOG_TARGET_COLUMN = "LOG_HOURS_PER_NM"
DEFAULT_CUTOFF = "2025-01-01"

BASE_CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "VESSEL_TYPE_ID",
    "HAS_CANAL_PASSAGE",
    "MONTH_NO",
)
SEASON_FEATURE = "SEASON"

DERIVED_CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "VESSEL_TYPE_CANAL_KEY",
    "VESSEL_TYPE_MONTH_KEY",
)

NUMERIC_FEATURES: Tuple[str, ...] = (
    "BALLAST_FRAC",
    "MILES_TOTAL",
    "MILES_BALLAST",
    "MILES_LOADED",
    "DAYS_AT_SEA",
    "DWT_SUMMER",
    "DRAFT_SUMMER",
    "LOA",
    "BEAM",
    "TOTAL_CARGO_QUANTITY",
    "N_CARGO_ROWS",
    "N_UNIQUE_CARGO",
    "VESSEL_AGE_YEARS",
    "CANAL_COST",
)


@dataclass
class PreparedSplits:
    train_path: Path
    val_path: Path
    metadata_path: Path
    train_size: int
    val_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare sailing-time ML dataset with chronological train/validation splits."
    )
    parser.add_argument(
        "--customer",
        "-c",
        help="Customer slug (default: northernlights). Use --list-customers to see options.",
    )
    parser.add_argument(
        "--cutoff-date",
        default=DEFAULT_CUTOFF,
        help="ISO date (YYYY-MM-DD) separating train (< cutoff) and validation (>= cutoff).",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        help="Optional validation fraction (0 < r < 1). Overrides --cutoff-date when provided while preserving chronological ordering.",
    )
    parser.add_argument(
        "--no-season-feature",
        action="store_true",
        help="Exclude derived SEASON categorical feature from output dataset.",
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
    (paths.derived_dir / ML_SUBDIR).mkdir(parents=True, exist_ok=True)
    return paths


def load_training(paths: CustomerPaths) -> pd.DataFrame:
    training_path = paths.derived_dir / DERIVED_SUBDIR / TRAINING_FILENAME
    if not training_path.exists():
        raise FileNotFoundError(
            f"Sailing-time training dataset missing for '{paths.key}'. "
            f"Expected at {training_path}. Run Models/build_sailing_time_dataset.py first."
        )
    df = pd.read_csv(training_path)
    return df


def derive_season(month: float | int | None) -> str:
    if pd.isna(month):
        return "unknown"
    month_int = int(month)
    if month_int in (12, 1, 2):
        return "winter"
    if month_int in (3, 4, 5):
        return "spring"
    if month_int in (6, 7, 8):
        return "summer"
    if month_int in (9, 10, 11):
        return "autumn"
    return "unknown"


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    if "VOYAGE_START_DATE" in working.columns:
        voyage_start_series = working["VOYAGE_START_DATE"]
    else:
        voyage_start_series = working.get("VOYAGE_START_TS")
    working["VOYAGE_START_DATE"] = pd.to_datetime(voyage_start_series, errors="coerce")
    working["MONTH_NO"] = pd.to_numeric(working.get("MONTH_NO"), errors="coerce").astype("Int64")
    if working["MONTH_NO"].isna().any():
        working.loc[working["MONTH_NO"].isna(), "MONTH_NO"] = (
            working.loc[working["MONTH_NO"].isna(), "VOYAGE_START_DATE"].dt.month
        )
    working["MONTH_NO"] = pd.to_numeric(working["MONTH_NO"], errors="coerce").astype("Int64")

    working["HAS_CANAL_PASSAGE"] = (
        pd.to_numeric(working.get("HAS_CANAL_PASSAGE"), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    working["VESSEL_TYPE_ID"] = pd.to_numeric(working.get("VESSEL_TYPE_ID"), errors="coerce").astype("Int64")

    working["SEASON"] = working["MONTH_NO"].map(derive_season)
    working["VESSEL_TYPE_CANAL_KEY"] = (
        working["VESSEL_TYPE_ID"].astype("Int64").astype(str) + "__" + working["HAS_CANAL_PASSAGE"].astype(str)
    )
    working["VESSEL_TYPE_MONTH_KEY"] = (
        working["VESSEL_TYPE_ID"].astype("Int64").astype(str) + "__" + working["MONTH_NO"].fillna(-1).astype(int).astype(str)
    )

    for col in NUMERIC_FEATURES + (TARGET_COLUMN,):
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    working[LOG_TARGET_COLUMN] = np.log1p(working[TARGET_COLUMN])
    return working


def perform_time_split(
    df: pd.DataFrame, cutoff_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame, datetime]:
    cutoff_ts = pd.to_datetime(cutoff_date).to_pydatetime()
    voyage_dates = df["VOYAGE_START_DATE"]
    train_mask = (voyage_dates.notna() & (voyage_dates < cutoff_ts)) | voyage_dates.isna()
    train_df = df.loc[train_mask].reset_index(drop=True)
    val_df = df.loc[~train_mask].reset_index(drop=True)
    return train_df, val_df, cutoff_ts


def perform_ratio_split(
    df: pd.DataFrame, validation_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, datetime | None]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1 (exclusive).")

    df_sorted = df.sort_values("VOYAGE_START_DATE", na_position="first").reset_index(drop=True)
    n_total = len(df_sorted)
    if n_total < 2:
        raise ValueError("Not enough rows to create a train/validation split.")

    val_size = max(1, int(round(n_total * validation_ratio)))
    val_size = min(n_total - 1, val_size)
    split_idx = n_total - val_size

    train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
    val_df = df_sorted.iloc[split_idx:].reset_index(drop=True)

    cutoff_series = val_df["VOYAGE_START_DATE"].dropna()
    cutoff_ts = cutoff_series.iloc[0].to_pydatetime() if not cutoff_series.empty else None
    return train_df, val_df, cutoff_ts


def export_splits(
    paths: CustomerPaths,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cutoff: datetime | None,
    columns: List[str],
    categorical_features: List[str],
    derived_categorical_features: List[str],
    numeric_features: List[str],
    strategy: dict,
) -> PreparedSplits:
    ml_dir = paths.derived_dir / ML_SUBDIR
    train_path = ml_dir / "sailing_time_train.parquet"
    val_path = ml_dir / "sailing_time_validation.parquet"
    metadata_path = ml_dir / "sailing_time_features.json"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    base_features = set(categorical_features).union(derived_categorical_features).union(numeric_features)
    additional_features = [
        col
        for col in columns
        if col not in base_features
        and col not in {TARGET_COLUMN, LOG_TARGET_COLUMN, "VOYAGE_ID", "VOYAGE_START_DATE"}
    ]

    metadata = {
        "customer": paths.key,
        "split_strategy": strategy,
        "cutoff_date": cutoff.strftime("%Y-%m-%d") if cutoff else None,
        "target_column": TARGET_COLUMN,
        "log_target_column": LOG_TARGET_COLUMN,
        "categorical_features": [col for col in categorical_features if col in columns],
        "derived_categorical_features": [col for col in derived_categorical_features if col in columns],
        "numeric_features": [col for col in numeric_features if col in columns],
        "additional_features": additional_features,
        "train_path": str(train_path.relative_to(ROOT)),
        "validation_path": str(val_path.relative_to(ROOT)),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return PreparedSplits(
        train_path=train_path,
        val_path=val_path,
        metadata_path=metadata_path,
        train_size=len(train_df),
        val_size=len(val_df),
    )


def main() -> None:
    args = parse_args()
    if args.list_customers:
        print(describe_customers())
        return

    paths = configure_customer(args.customer)
    raw_training = load_training(paths)
    enriched = enrich_features(raw_training)

    categorical_features = list(BASE_CATEGORICAL_FEATURES)
    if not args.no_season_feature:
        categorical_features.append(SEASON_FEATURE)

    derived_categorical_features = list(DERIVED_CATEGORICAL_FEATURES)
    numeric_features = list(NUMERIC_FEATURES)

    selected_columns = categorical_features + derived_categorical_features + numeric_features + [
        "VOYAGE_ID",
        "VOYAGE_START_DATE",
        TARGET_COLUMN,
        LOG_TARGET_COLUMN,
    ]

    missing = [col for col in selected_columns if col not in enriched.columns]
    if missing:
        raise ValueError(f"Expected columns missing from sailing-time dataset: {', '.join(sorted(missing))}")

    dataset = enriched[selected_columns]

    if args.validation_ratio is not None:
        train_df, val_df, cutoff_ts = perform_ratio_split(dataset, args.validation_ratio)
        split_strategy = {
            "type": "ratio",
            "validation_ratio": args.validation_ratio,
            "n_train": int(len(train_df)),
            "n_validation": int(len(val_df)),
        }
    else:
        train_df, val_df, cutoff_ts = perform_time_split(dataset, args.cutoff_date)
        split_strategy = {
            "type": "cutoff_date",
            "cutoff": args.cutoff_date,
            "n_train": int(len(train_df)),
            "n_validation": int(len(val_df)),
        }

    splits = export_splits(
        paths,
        train_df,
        val_df,
        cutoff_ts,
        selected_columns.copy(),
        categorical_features,
        derived_categorical_features,
        numeric_features,
        split_strategy,
    )

    print(f"[{paths.key}] Prepared sailing-time ML dataset")
    print(f"  Training rows:   {splits.train_size:,} -> {splits.train_path.relative_to(ROOT)}")
    print(f"  Validation rows: {splits.val_size:,} -> {splits.val_path.relative_to(ROOT)}")
    print(f"  Metadata:        {splits.metadata_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
