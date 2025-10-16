from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.customer_paths import (
    CustomerPaths,
    describe_customers,
    ensure_customer_dirs,
    resolve_customer,
)
_DEFAULT_PATHS = resolve_customer(None)
DER = _DEFAULT_PATHS.derived_dir
ensure_customer_dirs(_DEFAULT_PATHS)
TRAINING_PATH = DER / "port_turnaround_training.csv"
DAYS_IN_PORT_LOWER = 0.04
DAYS_IN_PORT_UPPER = 10.0
GROUP_TRIM_KEYS: Sequence[str] = ("PORT_ID", "TERMINAL_ID", "IS_BALLAST")
HIERARCHY: Sequence[Sequence[str]] = (
    ("PORT_ID", "TERMINAL_ID", "IS_BALLAST", "VESSEL_TYPE_ID", "MONTH_NO"),
    ("PORT_ID", "TERMINAL_ID", "IS_BALLAST", "VESSEL_TYPE_ID"),
    ("PORT_ID", "TERMINAL_ID", "IS_BALLAST"),
    ("PORT_ID", "IS_BALLAST"),
    ("PORT_ID",),
    (),  # global safety net
)
KEY_COLS = ["PORT_ID", "TERMINAL_ID", "IS_BALLAST", "VESSEL_TYPE_ID", "MONTH_NO"]


def configure_customer(slug: str | None) -> CustomerPaths:
    """Repoint global directories to the selected customer."""
    global DER, TRAINING_PATH
    paths = resolve_customer(slug)
    ensure_customer_dirs(paths)
    DER = paths.derived_dir
    TRAINING_PATH = DER / "port_turnaround_training.csv"
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct the port turnaround lookup table.")
    parser.add_argument(
        "--customer",
        "-c",
        help="Customer slug (e.g. 'northernlights', 'stena'). Defaults to northernlights.",
    )
    parser.add_argument(
        "--list-customers",
        action="store_true",
        help="Print available customer identifiers and exit.",
    )
    return parser.parse_args()


def try_write_parquet(df: pd.DataFrame, path: Path) -> bool:
    """Best-effort parquet export that degrades gracefully when engines are missing."""
    try:
        df.to_parquet(path, index=False)
        return True
    except (ImportError, ValueError) as exc:
        print(
            f"Skipping parquet output ({path.name}): install 'pyarrow' or 'fastparquet' to enable parquet exports."
        )
        if isinstance(exc, ValueError):
            print(f"  Reason: {exc}")
        return False


def load_training() -> pd.DataFrame:
    if not TRAINING_PATH.exists():
        raise FileNotFoundError(
            "Derived dataset missing. Run Models/build_port_turnaround_dataset.py first."
        )

    df = pd.read_csv(TRAINING_PATH)
    required = set(KEY_COLS + ["DAYS_IN_PORT"])
    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise ValueError(
            "Derived dataset lacks required columns: " + ", ".join(missing)
        )

    # Normalise dtypes so grouping is stable.
    numeric_cols = [col for col in required if col != "TERMINAL_ID"] + ["TERMINAL_ID"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["DAYS_IN_PORT"].between(DAYS_IN_PORT_LOWER, DAYS_IN_PORT_UPPER)]
    return df


def trim_extremes(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [col for col in GROUP_TRIM_KEYS if col in df.columns]
    if not group_cols:
        return df

    grouped = df.groupby(group_cols, dropna=False)["DAYS_IN_PORT"]
    q05 = grouped.transform(lambda s: s.quantile(0.05))
    q95 = grouped.transform(lambda s: s.quantile(0.95))
    mask = (df["DAYS_IN_PORT"] >= q05) & (df["DAYS_IN_PORT"] <= q95)
    mask |= q05.isna() | q95.isna()
    return df[mask].copy()


def aggregate_level(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    if not keys:  # global fallback row
        stats = {
            "median_days_in_port": df["DAYS_IN_PORT"].median(),
            "p10": df["DAYS_IN_PORT"].quantile(0.10),
            "p90": df["DAYS_IN_PORT"].quantile(0.90),
            "n_obs": df["DAYS_IN_PORT"].size,
        }
        row = {col: np.nan for col in KEY_COLS}
        row.update(stats)
        return pd.DataFrame([row])

    grouped = (
        df.groupby(list(keys), dropna=False)
        .agg(
            median_days_in_port=("DAYS_IN_PORT", "median"),
            p10=("DAYS_IN_PORT", lambda s: s.quantile(0.10)),
            p90=("DAYS_IN_PORT", lambda s: s.quantile(0.90)),
            n_obs=("DAYS_IN_PORT", "size"),
        )
        .reset_index()
    )
    missing_cols = [col for col in KEY_COLS if col not in grouped.columns]
    for col in missing_cols:
        grouped[col] = np.nan
    return grouped[KEY_COLS + ["median_days_in_port", "p10", "p90", "n_obs"]]


def specificity_score(row: pd.Series) -> int:
    score = 0
    score += 8 if not pd.isna(row["PORT_ID"]) else 0
    score += 4 if not pd.isna(row["TERMINAL_ID"]) else 0
    score += 2 if not pd.isna(row["IS_BALLAST"]) else 0
    score += 1 if not pd.isna(row["VESSEL_TYPE_ID"]) else 0
    score += 1 if not pd.isna(row["MONTH_NO"]) else 0
    return score


def main() -> None:
    args = parse_args()
    if args.list_customers:
        print(describe_customers())
        return
    paths = configure_customer(args.customer)
    training = load_training()
    training_trimmed = trim_extremes(training)

    frames = []
    for level, keys in enumerate(HIERARCHY, start=1):
        aggregated = aggregate_level(training_trimmed, keys)
        aggregated["level"] = level
        frames.append(aggregated)

    lookup_all = pd.concat(frames, ignore_index=True)

    # Enforce minimum support except for the global fallback level.
    max_level = len(HIERARCHY)
    lookup_all = lookup_all[(lookup_all["n_obs"] >= 3) | (lookup_all["level"] == max_level)]
    lookup_all["median_days_in_port"] = lookup_all["median_days_in_port"].clip(
        lower=DAYS_IN_PORT_LOWER, upper=DAYS_IN_PORT_UPPER
    )
    lookup_all["specificity_score"] = lookup_all.apply(specificity_score, axis=1)

    lookup_all.sort_values(
        by=["specificity_score", "n_obs"],
        ascending=[False, False],
        inplace=True,
    )
    lookup = lookup_all.drop_duplicates(subset=KEY_COLS, keep="first").reset_index(drop=True)

    output_cols = KEY_COLS + ["median_days_in_port", "n_obs", "specificity_score", "level"]
    try_write_parquet(lookup[output_cols], DER / "port_turnaround_lookup.parquet")
    lookup[output_cols].to_csv(DER / "port_turnaround_lookup.csv", index=False)

    qa_bins = (
        training_trimmed.assign(
            BIN=lambda d: pd.cut(
                d["DAYS_IN_PORT"], bins=[0, 0.5, 1, 2, 3, 5, 10], include_lowest=True
            )
        )
        .groupby("BIN", observed=False)
        .size()
        .reset_index(name="n")
    )
    qa_bins.to_csv(DER / "port_turnaround_lookup_qa_bins.csv", index=False)

    lookup_csv = DER / "port_turnaround_lookup.csv"
    qa_bins_csv = DER / "port_turnaround_lookup_qa_bins.csv"

    print(f"[{paths.key}] wrote {len(lookup):,} rows -> {lookup_csv.relative_to(ROOT)}")
    if (DER / "port_turnaround_lookup.parquet").exists():
        print(f"[{paths.key}] parquet -> {(DER / 'port_turnaround_lookup.parquet').relative_to(ROOT)}")
    print(f"[{paths.key}] QA bins -> {qa_bins_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
