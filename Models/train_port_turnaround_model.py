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

CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "PORT_ID",
    "TERMINAL_ID",
    "IS_BALLAST",
    "VESSEL_TYPE_ID",
    "MONTH_NO",
    "HAS_CANAL_PASSAGE",
    "COMMODITY_GROUP_ID",
    "COMMODITY_CODE",
)

DERIVED_CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "PORT_TERMINAL_KEY",
    "PORT_IS_BALLAST_KEY",
) 

NUMERIC_FEATURES: Tuple[str, ...] = (
    "DWT_SUMMER",
    "DRAFT_SUMMER",
    "LOA",
    "BEAM",
    "DAYS_STOPPAGES",
    "DAYS_EXTRA_IN_PORT",
)

DERIVED_NUMERIC_FEATURES: Tuple[str, ...] = (
    "PORT_MEDIAN_DAYS",
    "PORT_IS_BALLAST_MEDIAN_DAYS",
)

ML_SUBDIR = "ML"
VOYAGES_EXPORT_NAME = "voyages_completed_asof_2025-12-31.csv"
TARGET_COLUMN = "DAYS_IN_PORT"
LOG_TARGET_COLUMN = "LOG_DAYS_IN_PORT"
DEFAULT_CUTOFF = "2025-01-01"


@dataclass
class PreparedSplits:
    train_path: Path
    val_path: Path
    metadata_path: Path
    train_size: int
    val_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare feature dataset and time-based splits for port turnaround ML modelling."
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
        help="Optional validation fraction (0 < r < 1). Uses chronological order and overrides --cutoff-date when provided.",
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
    ml_dir = paths.derived_dir / ML_SUBDIR
    ml_dir.mkdir(parents=True, exist_ok=True)
    return paths


def load_training(paths: CustomerPaths) -> pd.DataFrame:
    training_path = paths.derived_dir / "port_turnaround_training.csv"
    if not training_path.exists():
        raise FileNotFoundError(
            f"Derived training dataset missing for customer '{paths.key}'. "
            "Run Models/build_port_turnaround_dataset.py first."
        )
    df = pd.read_csv(training_path)
    return df


def load_voyage_dates(paths: CustomerPaths) -> pd.DataFrame:
    voyages_path = paths.raw_dir / VOYAGES_EXPORT_NAME
    if not voyages_path.exists():
        raise FileNotFoundError(
            f"Voyage export '{VOYAGES_EXPORT_NAME}' missing under {paths.raw_dir}. "
            "Ensure raw extracts are staged before preparing the ML dataset."
        )
    voyages = pd.read_csv(voyages_path)
    if "VOYAGE_ID" not in voyages.columns:
        raise ValueError("Voyages export lacks VOYAGE_ID column.")

    date_col = None
    for candidate in ("VOYAGE_START_TS", "ESTIMATED_VOYAGE_START_DATE"):
        if candidate in voyages.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError("Voyages export does not contain a voyage start date column.")

    voyages["VOYAGE_START_DATE"] = pd.to_datetime(
        voyages[date_col], errors="coerce", format="%d-%b-%y"
    )
    return voyages[["VOYAGE_ID", "VOYAGE_START_DATE"]]


def enrich_features(df: pd.DataFrame, voyages: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(voyages, on="VOYAGE_ID", how="left")
    # Derived categorical keys
    terminal_filled = pd.to_numeric(merged["TERMINAL_ID"], errors="coerce").fillna(-1).astype("Int64")
    port_ids = pd.to_numeric(merged["PORT_ID"], errors="coerce").astype("Int64")
    ballast_flags = pd.to_numeric(merged["IS_BALLAST"], errors="coerce").fillna(-1).astype(int)
    merged["PORT_TERMINAL_KEY"] = port_ids.astype(str) + "__" + terminal_filled.astype(str)
    merged["PORT_IS_BALLAST_KEY"] = port_ids.astype(str) + "__" + ballast_flags.astype(str)

    merged["PORT_ID"] = port_ids
    merged["TERMINAL_ID"] = terminal_filled.mask(terminal_filled == -1, pd.NA)
    merged["IS_BALLAST"] = ballast_flags
    merged["VESSEL_TYPE_ID"] = pd.to_numeric(merged["VESSEL_TYPE_ID"], errors="coerce").astype("Int64")
    month_no = pd.to_numeric(merged["MONTH_NO"], errors="coerce")
    if "VOYAGE_START_DATE" in merged.columns:
        month_from_date = merged["VOYAGE_START_DATE"].dt.month
        month_no = month_no.fillna(month_from_date)
    merged["MONTH_NO"] = month_no.astype("Int64")
    merged["HAS_CANAL_PASSAGE"] = (
        pd.to_numeric(merged.get("HAS_CANAL_PASSAGE"), errors="coerce").fillna(-1).astype(int)
    )

    # Ensure categorical columns as string for downstream encoders (but keep original numeric values too)
    for cat in ("TERMINAL_ID", "VESSEL_TYPE_ID", "COMMODITY_GROUP_ID"):
        if cat in merged.columns:
            merged[cat] = merged[cat].astype("float64")
    merged["COMMODITY_CODE"] = (
        merged.get("COMMODITY_CODE", pd.Series(index=merged.index))
        .fillna("MISSING")
        .astype(str)
    )

    # Numeric sanitisation
    for col in NUMERIC_FEATURES:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Aggregate features across full modelling view (acts as strong priors)
    port_median = merged.groupby("PORT_ID")[TARGET_COLUMN].median()
    merged["PORT_MEDIAN_DAYS"] = merged["PORT_ID"].map(port_median)

    port_ballast_median = merged.groupby(["PORT_ID", "IS_BALLAST"])[TARGET_COLUMN].median().to_dict()
    merged["PORT_IS_BALLAST_MEDIAN_DAYS"] = [
        port_ballast_median.get((pid, ballast), np.nan)
        for pid, ballast in zip(merged["PORT_ID"], merged["IS_BALLAST"])
    ]
    merged["PORT_IS_BALLAST_MEDIAN_DAYS"] = merged["PORT_IS_BALLAST_MEDIAN_DAYS"].fillna(
        merged["PORT_MEDIAN_DAYS"]
    )
    merged["PORT_MEDIAN_DAYS"] = merged["PORT_MEDIAN_DAYS"].fillna(merged[TARGET_COLUMN].median())
    merged["PORT_IS_BALLAST_MEDIAN_DAYS"] = merged["PORT_IS_BALLAST_MEDIAN_DAYS"].fillna(
        merged[TARGET_COLUMN].median()
    )

    merged[LOG_TARGET_COLUMN] = np.log1p(merged[TARGET_COLUMN])
    return merged


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
    strategy: dict,
) -> PreparedSplits:
    ml_dir = paths.derived_dir / ML_SUBDIR
    train_path = ml_dir / "port_turnaround_train.parquet"
    val_path = ml_dir / "port_turnaround_validation.parquet"
    metadata_path = ml_dir / "port_turnaround_features.json"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    base_feature_set = (
        set(CATEGORICAL_FEATURES)
        .union(DERIVED_CATEGORICAL_FEATURES)
        .union(NUMERIC_FEATURES)
        .union(DERIVED_NUMERIC_FEATURES)
    )
    derived_features = [
        col
        for col in columns
        if col not in base_feature_set
        and col not in {TARGET_COLUMN, LOG_TARGET_COLUMN, "VOYAGE_ID", "VOYAGE_START_DATE"}
    ]

    metadata = {
        "customer": paths.key,
        "split_strategy": strategy,
        "cutoff_date": cutoff.strftime("%Y-%m-%d") if cutoff else None,
        "target_column": TARGET_COLUMN,
        "log_target_column": LOG_TARGET_COLUMN,
        "categorical_features": list(CATEGORICAL_FEATURES),
        "derived_categorical_features": list(DERIVED_CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "derived_numeric_features": list(DERIVED_NUMERIC_FEATURES),
        "additional_features": derived_features,
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
    voyage_dates = load_voyage_dates(paths)
    enriched = enrich_features(raw_training, voyage_dates)

    selected_columns = list(
        CATEGORICAL_FEATURES
        + DERIVED_CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
        + DERIVED_NUMERIC_FEATURES
    ) + [
        TARGET_COLUMN,
        LOG_TARGET_COLUMN,
        "VOYAGE_ID",
        "VOYAGE_START_DATE",
    ]
    missing_cols = [col for col in selected_columns if col not in enriched.columns]
    if missing_cols:
        raise ValueError(f"Expected columns missing from enriched dataset: {', '.join(missing_cols)}")

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

    splits = export_splits(paths, train_df, val_df, cutoff_ts, selected_columns, split_strategy)

    print(f"[{paths.key}] Prepared ML feature dataset")
    print(f"  Training rows:   {splits.train_size:,} -> {splits.train_path.relative_to(ROOT)}")
    print(f"  Validation rows: {splits.val_size:,} -> {splits.val_path.relative_to(ROOT)}")
    print(f"  Metadata:        {splits.metadata_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
