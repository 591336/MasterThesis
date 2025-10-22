from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

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

DERIVED_SUBDIR = "sailing_time"
TRAINING_FILENAME = "sailing_time_training.csv"
LOOKUP_FILENAME = "sailing_time_lookup.csv"
LOOKUP_PARQUET = "sailing_time_lookup.parquet"
QA_BINS_FILENAME = "sailing_time_lookup_qa_bins.csv"

HOURS_PER_NM_LOWER = 0.02
HOURS_PER_NM_UPPER = 5.0
MIN_SUPPORT = 3

HIERARCHY: Sequence[Sequence[str]] = (
    ("VESSEL_TYPE_ID", "HAS_CANAL_PASSAGE", "MONTH_NO"),
    ("VESSEL_TYPE_ID", "HAS_CANAL_PASSAGE"),
    ("VESSEL_TYPE_ID",),
    (),
)
KEY_COLUMNS = ["VESSEL_TYPE_ID", "HAS_CANAL_PASSAGE", "MONTH_NO"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sailing time lookup table.")
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
    return parser.parse_args()


def configure_paths(customer_slug: str | None) -> tuple[pd.DataFrame, Path]:
    paths = resolve_customer(customer_slug)
    ensure_customer_dirs(paths)

    derived_dir = paths.derived_dir / DERIVED_SUBDIR
    training_path = derived_dir / TRAINING_FILENAME
    if not training_path.exists():
        raise FileNotFoundError(
            f"Training dataset missing for '{paths.key}'. Expected at {training_path}. "
            "Run Models/build_sailing_time_dataset.py first."
        )
    df = pd.read_csv(training_path)
    return df, derived_dir


def aggregate_level(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    if not keys:
        stats = {
            "median_hours_per_nm": df["HOURS_PER_NM"].median(),
            "p10": df["HOURS_PER_NM"].quantile(0.10),
            "p90": df["HOURS_PER_NM"].quantile(0.90),
            "n_obs": df["HOURS_PER_NM"].size,
        }
        row = {col: np.nan for col in KEY_COLUMNS}
        row.update(stats)
        return pd.DataFrame([row])

    grouped = (
        df.groupby(list(keys), dropna=False)["HOURS_PER_NM"]
        .agg(
            median_hours_per_nm="median",
            p10=lambda s: s.quantile(0.10),
            p90=lambda s: s.quantile(0.90),
            n_obs="size",
        )
        .reset_index()
    )
    missing_cols = [col for col in KEY_COLUMNS if col not in grouped.columns]
    for col in missing_cols:
        grouped[col] = np.nan
    return grouped[KEY_COLUMNS + ["median_hours_per_nm", "p10", "p90", "n_obs"]]


def compute_lookup(df: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for level, keys in enumerate(HIERARCHY, start=1):
        agg = aggregate_level(df, keys)
        agg["level"] = level
        frames.append(agg)

    lookup_all = pd.concat(frames, ignore_index=True)
    max_level = len(HIERARCHY)
    lookup_all = lookup_all[
        (lookup_all["n_obs"] >= MIN_SUPPORT) | (lookup_all["level"] == max_level)
    ]
    lookup_all["median_hours_per_nm"] = lookup_all["median_hours_per_nm"].clip(
        lower=HOURS_PER_NM_LOWER,
        upper=HOURS_PER_NM_UPPER,
    )

    specificity = (
        lookup_all["VESSEL_TYPE_ID"].notna().astype(int) * 4
        + lookup_all["HAS_CANAL_PASSAGE"].notna().astype(int) * 2
        + lookup_all["MONTH_NO"].notna().astype(int)
    )
    lookup_all["specificity_score"] = specificity

    lookup_all.sort_values(
        by=["specificity_score", "n_obs"],
        ascending=[False, False],
        inplace=True,
    )
    lookup = lookup_all.drop_duplicates(subset=KEY_COLUMNS, keep="first").reset_index(drop=True)
    return lookup


def write_outputs(lookup: pd.DataFrame, derived_dir: Path) -> None:
    lookup_csv = derived_dir / LOOKUP_FILENAME
    lookup_parquet = derived_dir / LOOKUP_PARQUET
    lookup.to_csv(lookup_csv, index=False)
    try:
        lookup.to_parquet(lookup_parquet, index=False)
    except (ImportError, ValueError):
        print("Parquet export skipped (install 'pyarrow' or 'fastparquet').")

    qa_bins = (
        lookup.assign(
            BIN=lambda d: pd.cut(
                d["median_hours_per_nm"],
                bins=[0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
                include_lowest=True,
            )
        )
        .groupby("BIN", observed=False)
        .size()
        .reset_index(name="n")
    )
    qa_bins.to_csv(derived_dir / QA_BINS_FILENAME, index=False)

    print(f"Wrote lookup -> {lookup_csv} ({len(lookup):,} rows)")
    print(f"QA bins -> {derived_dir / QA_BINS_FILENAME}")


def main() -> None:
    args = parse_args()
    if args.list_customers:
        print(describe_customers())
        return

    df, derived_dir = configure_paths(args.customer)
    df = df[df["HOURS_PER_NM"].between(HOURS_PER_NM_LOWER, HOURS_PER_NM_UPPER)]
    lookup = compute_lookup(df)
    write_outputs(lookup, derived_dir)


if __name__ == "__main__":
    main()
