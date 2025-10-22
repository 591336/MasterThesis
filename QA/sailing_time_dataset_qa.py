from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

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

DERIVED_SUBDIR = "sailing_time"
TRAIN_CSV_NAME = "sailing_time_training.csv"
QA_DIRNAME = "QA"
FIG_DIRNAME = "figures"
REPORTS_DIRNAME = "reports"
TABLES_DIRNAME = "tables"

HOURS_PER_NM_LOWER = 0.02
HOURS_PER_NM_UPPER = 5.0

KEY_COLUMNS = [
    "VESSEL_TYPE_ID",
    "HAS_CANAL_PASSAGE",
    "MONTH_NO",
    "BALLAST_FRAC",
    "DWT_SUMMER",
]

GROUP_COLUMNS = ["VESSEL_TYPE_ID", "HAS_CANAL_PASSAGE", "MONTH_NO"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA suite for sailing time training dataset.")
    parser.add_argument(
        "--customer",
        "-c",
        help="Customer slug (default northernlights). Use --list-customers to see options.",
    )
    parser.add_argument(
        "--list-customers",
        action="store_true",
        help="Print available customers and exit.",
    )
    return parser.parse_args()


def configure_paths(customer_slug: str | None) -> tuple[pd.DataFrame, dict[str, Path]]:
    paths = resolve_customer(customer_slug)
    ensure_customer_dirs(paths)

    derived_dir = paths.derived_dir / DERIVED_SUBDIR
    training_csv = derived_dir / TRAIN_CSV_NAME
    if not training_csv.exists():
        raise FileNotFoundError(
            f"Training dataset missing for '{paths.key}'. Expected at {training_csv}. "
            "Run Models/build_sailing_time_dataset.py first."
        )

    df = pd.read_csv(training_csv)
    qa_dir = derived_dir / QA_DIRNAME
    reports_dir = qa_dir / REPORTS_DIRNAME
    tables_dir = qa_dir / TABLES_DIRNAME
    figures_dir = qa_dir / FIG_DIRNAME

    for directory in (qa_dir, reports_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    locations = {
        "reports": reports_dir,
        "tables": tables_dir,
        "figures": figures_dir,
    }

    return df, locations


def write_overview(df: pd.DataFrame, reports_dir: Path) -> None:
    overview_path = reports_dir / "sailing_time_training_overview.txt"
    lines = [
        "Sailing time training dataset - overview",
        f"Rows: {len(df):,}",
        "",
        "Column dtypes:",
    ]
    lines.extend(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
    lines.extend(["", "Null ratios (selected columns):"])
    for col in KEY_COLUMNS:
        if col in df:
            ratio = df[col].isna().mean()
            lines.append(f"  {col}: {ratio:.2%}")
        else:
            lines.append(f"  {col}: column missing")
    overview_path.write_text("\n".join(lines), encoding="utf-8")


def write_guardrail_stats(df: pd.DataFrame, reports_dir: Path) -> None:
    path = reports_dir / "sailing_time_training_guardrails.txt"
    if "HOURS_PER_NM" not in df:
        path.write_text("HOURS_PER_NM column missing", encoding="utf-8")
        return

    series = pd.to_numeric(df["HOURS_PER_NM"], errors="coerce").dropna()
    lines = [
        "HOURS_PER_NM guardrail check",
        f"Observations (non-null): {len(series):,}",
        "",
    ]
    if series.empty:
        lines.append("No non-null values available.")
    else:
        stats = series.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        lines.append("Key percentiles (hours per NM):")
        lines.append(f"  min:  {stats['min']:.4f}")
        lines.append(f"  p05:  {stats['5%']:.4f}")
        lines.append(f"  p25:  {stats['25%']:.4f}")
        lines.append(f"  median: {stats['50%']:.4f}")
        lines.append(f"  p75:  {stats['75%']:.4f}")
        lines.append(f"  p95:  {stats['95%']:.4f}")
        lines.append(f"  max:  {stats['max']:.4f}")
        below = int((series < HOURS_PER_NM_LOWER).sum())
        above = int((series > HOURS_PER_NM_UPPER).sum())
        lines.append("")
        lines.append(f"Guardrail: {HOURS_PER_NM_LOWER} <= HOURS_PER_NM <= {HOURS_PER_NM_UPPER}")
        lines.append(f"  Below lower bound: {below}")
        lines.append(f"  Above upper bound: {above}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_missingness(df: pd.DataFrame, group_cols: List[str], available: List[str]) -> pd.DataFrame:
    indicator = df[available].isna().astype(float).add_suffix("_missing_ratio")
    combined = pd.concat([df[group_cols], indicator], axis=1)
    grouped = combined.groupby(group_cols, dropna=False)
    summary = grouped.mean(numeric_only=True).reset_index()
    summary["n_obs"] = grouped.size().to_numpy()
    summary.sort_values("n_obs", ascending=False, inplace=True)
    return summary


def write_missingness_tables(df: pd.DataFrame, tables_dir: Path) -> List[Path]:
    available_keys = [col for col in KEY_COLUMNS if col in df.columns]
    outputs: List[Path] = []
    if not available_keys:
        empty_path = tables_dir / "sailing_time_missingness_by_vessel_type.csv"
        empty_path.write_text("Required columns missing for missingness summary", encoding="utf-8")
        outputs.append(empty_path)
        return outputs

    by_vessel_type = summarize_missingness(df, ["VESSEL_TYPE_ID"], available_keys)
    vt_path = tables_dir / "sailing_time_missingness_by_vessel_type.csv"
    by_vessel_type.to_csv(vt_path, index=False)
    outputs.append(vt_path)

    if {"VESSEL_TYPE_ID", "HAS_CANAL_PASSAGE"} <= set(df.columns):
        by_type_canal = summarize_missingness(df, ["VESSEL_TYPE_ID", "HAS_CANAL_PASSAGE"], available_keys)
        tc_path = tables_dir / "sailing_time_missingness_by_type_canal.csv"
        by_type_canal.to_csv(tc_path, index=False)
        outputs.append(tc_path)

    return outputs


def write_missingness_notes(df: pd.DataFrame, tables: List[Path], reports_dir: Path) -> None:
    path = reports_dir / "sailing_time_missingness_notes.txt"
    available_keys = [col for col in KEY_COLUMNS if col in df.columns]
    lines = [
        "Missingness hotspots",
        f"Threshold: 50% missing",
        "",
    ]
    if not available_keys:
        lines.append("No key columns available for missingness audit.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    any_hotspots = False
    threshold = 0.50
    max_listed = 5

    for table_path in tables:
        if not table_path.exists():
            continue
        summary = pd.read_csv(table_path)
        ratio_cols = [col for col in summary.columns if col.endswith("_missing_ratio")]
        if not ratio_cols:
            continue
        for ratio_col in ratio_cols:
            col = ratio_col.replace("_missing_ratio", "")
            lines.append(f"Column: {col} ({table_path.name})")
            hotspots = summary[summary[ratio_col] >= threshold]
            if hotspots.empty:
                lines.append("  - No groups exceed threshold")
                continue
            any_hotspots = True
            listed = hotspots.sort_values(ratio_col, ascending=False).head(max_listed)
            for _, row in listed.iterrows():
                group_desc = ", ".join(f"{k}={row.get(k)}" for k in summary.columns if k not in ratio_cols + ["n_obs"])
                ratio = row[ratio_col]
                n_obs = int(row.get("n_obs", 0))
                lines.append(f"  - [{group_desc}] {ratio:.0%} missing (n={n_obs})")
            lines.append("")

    if not any_hotspots:
        lines.append("No hotspots above threshold detected.")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_histogram(df: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "sailing_time_hours_per_nm_hist.csv"
    if "HOURS_PER_NM" not in df:
        path.write_text("HOURS_PER_NM column missing", encoding="utf-8")
        return path
    hist_counts, bin_edges = np.histogram(df["HOURS_PER_NM"].dropna(), bins=20, range=(0, HOURS_PER_NM_UPPER))
    hist_df = pd.DataFrame({"bin_left": bin_edges[:-1], "bin_right": bin_edges[1:], "count": hist_counts})
    hist_df.to_csv(path, index=False)
    return path


def write_group_counts(df: pd.DataFrame, tables_dir: Path) -> Path:
    group_cols = [col for col in GROUP_COLUMNS if col in df.columns]
    path = tables_dir / "sailing_time_group_counts.csv"
    if not group_cols:
        path.write_text("Required grouping columns missing.", encoding="utf-8")
        return path
    summary = (
        df.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_obs")
        .sort_values("n_obs", ascending=False)
    )
    summary.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    if args.list_customers:
        print(describe_customers())
        return

    df, locations = configure_paths(args.customer)
    write_overview(df, locations["reports"])
    write_guardrail_stats(df, locations["reports"])
    missing_tables = write_missingness_tables(df, locations["tables"])
    write_missingness_notes(df, missing_tables, locations["reports"])
    write_histogram(df, locations["figures"])
    write_group_counts(df, locations["tables"])

    print("Generated sailing time QA artefacts under", locations["reports"].parent)


if __name__ == "__main__":
    main()
