from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.customer_paths import (  # noqa: E402
    CustomerPaths,
    ensure_customer_dirs,
    resolve_customer,
)

RAW_VOYAGES = "sailing_time_voyages_asof_2025-12-31.csv"
RAW_VESSELS = "sailing_time_vessels_reference.csv"
RAW_CARGO_COUNTS = "sailing_time_voyage_cargo_counts.csv"

DERIVED_FOLDER = "sailing_time"
TRAINING_FILENAME = "sailing_time_training.csv"
PARQUET_FILENAME = "sailing_time_training.parquet"
QA_COUNTS_FILENAME = "sailing_time_training_counts.csv"

HOURS_PER_NM_LOWER = 0.02  # roughly 50 knots upper speed (24/1200)
HOURS_PER_NM_UPPER = 5.0    # ~0.2 knots lower speed; trims extreme delays


@dataclass
class DatasetPaths:
    customer: str
    raw_dir: Path
    derived_dir: Path
    voyages_csv: Path
    vessels_csv: Path
    cargo_counts_csv: Optional[Path]
    output_csv: Path
    output_parquet: Path
    output_counts: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sailing-time training dataset (hours-per-nautical-mile)."
    )
    parser.add_argument(
        "--customer",
        "-c",
        help="Customer slug (default: northernlights). Use --list-customers for options.",
    )
    parser.add_argument(
        "--list-customers",
        action="store_true",
        help="Print available customers and exit.",
    )
    return parser.parse_args()


def configure_paths(customer_slug: str | None) -> DatasetPaths:
    paths = resolve_customer(customer_slug)
    ensure_customer_dirs(paths)

    derived_dir = paths.derived_dir / DERIVED_FOLDER
    derived_dir.mkdir(parents=True, exist_ok=True)

    voyages_csv = paths.raw_dir / RAW_VOYAGES
    vessels_csv = paths.raw_dir / RAW_VESSELS
    cargo_counts_csv = paths.raw_dir / RAW_CARGO_COUNTS
    if not cargo_counts_csv.exists():
        cargo_counts_csv = None

    return DatasetPaths(
        customer=paths.key,
        raw_dir=paths.raw_dir,
        derived_dir=derived_dir,
        voyages_csv=voyages_csv,
        vessels_csv=vessels_csv,
        cargo_counts_csv=cargo_counts_csv,
        output_csv=derived_dir / TRAINING_FILENAME,
        output_parquet=derived_dir / PARQUET_FILENAME,
        output_counts=derived_dir / QA_COUNTS_FILENAME,
    )


def load_voyages(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Voyage export missing: {path}")
    df = pd.read_csv(path)
    df.columns = [c.upper() for c in df.columns]
    df["VOYAGE_START_DATE"] = pd.to_datetime(df.get("VOYAGE_START_TS"), errors="coerce")
    return df


def load_vessels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Vessel reference missing: {path}")
    df = pd.read_csv(path)
    df.columns = [c.upper() for c in df.columns]
    # Drop obvious test vessels
    if "VESSEL_NAME" in df.columns:
        mask = df["VESSEL_NAME"].astype(str).str.contains("test", case=False, na=False)
        df = df.loc[~mask].copy()
    return df


def load_cargo_counts(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["VOYAGE_ID", "N_CARGO_ROWS", "N_UNIQUE_CARGO", "TOTAL_CARGO_QUANTITY"])
    df = pd.read_csv(path)
    df.columns = [c.upper() for c in df.columns]
    return df


def compute_hours_per_nm(row: pd.Series) -> float:
    distance = row.get("MILES_TOTAL", np.nan)
    days_at_sea = row.get("DAYS_AT_SEA", np.nan)
    if pd.isna(distance) or distance <= 0:
        return np.nan
    if pd.isna(days_at_sea) or days_at_sea <= 0:
        return np.nan
    return float(days_at_sea * 24.0 / distance)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["HOURS_PER_NM"] = df.apply(compute_hours_per_nm, axis=1)
    df["BALLAST_FRAC"] = np.where(
        df["MILES_TOTAL"] > 0,
        df["MILES_BALLAST"] / df["MILES_TOTAL"],
        np.nan,
    )
    df["MONTH_NO"] = df["VOYAGE_START_DATE"].dt.month
    df["YEAR"] = df["VOYAGE_START_DATE"].dt.year
    df["VOYAGE_YEAR_MONTH"] = df["VOYAGE_START_DATE"].dt.to_period("M").astype(str)
    df["HAS_CANAL_PASSAGE"] = pd.to_numeric(df.get("HAS_CANAL_PASSAGE"), errors="coerce").fillna(0).astype(int)
    df["CANAL_COST"] = pd.to_numeric(df.get("CANAL_COST"), errors="coerce")

    if "BUILT_YEAR" in df.columns:
        df["VESSEL_AGE_YEARS"] = df["YEAR"] - pd.to_numeric(df["BUILT_YEAR"], errors="coerce")

    numeric_cols = [
        "MILES_BALLAST",
        "MILES_LOADED",
        "MILES_TOTAL",
        "DAYS_AT_SEA",
        "DAYS_IN_PORT_TOTAL",
        "DAYS_TOTAL",
        "DWT_SUMMER",
        "DRAFT_SUMMER",
        "LOA",
        "BEAM",
        "TOTAL_CARGO_QUANTITY",
        "N_CARGO_ROWS",
        "N_UNIQUE_CARGO",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    mask_valid = df["HOURS_PER_NM"].between(HOURS_PER_NM_LOWER, HOURS_PER_NM_UPPER)
    filtered = df.loc[mask_valid].copy()
    return filtered


def write_outputs(df: pd.DataFrame, paths: DatasetPaths) -> None:
    df.to_csv(paths.output_csv, index=False)
    try:
        df.to_parquet(paths.output_parquet, index=False)
    except (ImportError, ValueError):
        print("Parquet export skipped (install 'pyarrow' or 'fastparquet' to enable).")

    group_cols = ["VESSEL_TYPE_ID", "HAS_CANAL_PASSAGE", "MONTH_NO"]
    available_group_cols = [c for c in group_cols if c in df.columns]
    if available_group_cols:
        counts = (
            df.groupby(available_group_cols, dropna=False)
            .size()
            .reset_index(name="n_obs")
            .sort_values("n_obs", ascending=False)
        )
    else:
        counts = pd.DataFrame({"n_obs": [len(df)]})
    counts.to_csv(paths.output_counts, index=False)


def log_summary(df: pd.DataFrame, paths: DatasetPaths) -> None:
    summary = {
        "customer": paths.customer,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "n_rows": int(len(df)),
        "hours_per_nm_min": float(df["HOURS_PER_NM"].min()),
        "hours_per_nm_p50": float(df["HOURS_PER_NM"].median()),
        "hours_per_nm_max": float(df["HOURS_PER_NM"].max()),
        "fraction_missing_hours": float(df["HOURS_PER_NM"].isna().mean()),
    }
    log_path = paths.derived_dir / "sailing_time_dataset_log.jsonl"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{summary}\n")


def main() -> None:
    args = parse_args()
    if args.list_customers:
        from utils.customer_paths import describe_customers  # noqa:E402

        print(describe_customers())
        return

    paths = configure_paths(args.customer)

    voyages = load_voyages(paths.voyages_csv)
    vessels = load_vessels(paths.vessels_csv)
    cargo_counts = load_cargo_counts(paths.cargo_counts_csv)

    df = voyages.merge(
        vessels[
            [
                "VESSEL_ID",
                "VESSEL_NAME",
                "VESSEL_TYPE_ID",
                "DWT_SUMMER",
                "DRAFT_SUMMER",
                "LOA",
                "BEAM",
                "BUILT_YEAR",
                "FLAG_ID",
            ]
        ],
        on=["VESSEL_ID", "VESSEL_TYPE_ID"],
        how="left",
    )

    if not cargo_counts.empty:
        df = df.merge(cargo_counts, on="VOYAGE_ID", how="left")

    df = engineer_features(df)
    filtered = filter_quality(df)

    write_outputs(filtered, paths)
    log_summary(filtered, paths)

    print(f"[{paths.customer}] Wrote {len(filtered):,} rows -> {paths.output_csv}")
    print(f"[{paths.customer}] QA counts -> {paths.output_counts}")


if __name__ == "__main__":
    main()
