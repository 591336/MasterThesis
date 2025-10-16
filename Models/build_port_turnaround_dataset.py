from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

# --------------------------------------------------------------------------------------
# Purpose
# -------
# Build a single, human-readable training dataset for port turnaround modeling.
# We only JOIN + CLEAN here. No modeling, no medians — that comes in step02b.
#
# Inputs (from your repo structure):
#   DataSets/Raw/NorthernLightsTest/
#     - port_calls_completed_asof_2025-12-31.csv
#     - voyages_completed_asof_2025-12-31.csv
#     - vessels_reference.csv
#     - cargo_asof_2025-12-31.csv        (optional; only for commodity enrichment)
#     - commodity_reference.csv          (optional; only for commodity enrichment)
#
# Outputs:
#   DataSets/Derived/
#     - port_turnaround_training.csv
#     - port_turnaround_training.parquet
#     - port_turnaround_training_counts.csv  (QA: counts per (port, terminal, ballast))
#
# Notes:
# - ARRIVAL/DEPARTURE columns in TBL_PORT_CALL are numeric flags in your DB, not timestamps,
#   so we don’t use them. Label = DAYS_IN_PORT.
# - We add MONTH_NO from the voyage start date (if present) as a seasonal feature.
# - We keep VESSEL_TYPE_ID (from voyages or vessels) and basic vessel size proxies.
# - We trim blatant outliers so downstream stats/models aren’t dominated by junk.
# --------------------------------------------------------------------------------------

_DEFAULT_PATHS = resolve_customer(None)
RAW = _DEFAULT_PATHS.raw_dir
DER = _DEFAULT_PATHS.derived_dir
ensure_customer_dirs(_DEFAULT_PATHS)


def configure_customer(slug: str | None) -> CustomerPaths:
    """Repoint global directories to the selected customer."""
    global RAW, DER
    paths = resolve_customer(slug)
    ensure_customer_dirs(paths)
    RAW = paths.raw_dir
    DER = paths.derived_dir
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build port turnaround training dataset.")
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


def load_csv(name: str) -> pd.DataFrame:
    """Load a CSV from RAW and upper-case the column names for easier joins."""
    df = pd.read_csv(RAW / name)
    df.columns = [c.upper() for c in df.columns]
    return df


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


def main() -> None:
    args = parse_args()
    if args.list_customers:
        print(describe_customers())
        return
    paths = configure_customer(args.customer)

    # ---- Load mandatory inputs ----
    pc = load_csv("port_calls_completed_asof_2025-12-31.csv")
    voy = load_csv("voyages_completed_asof_2025-12-31.csv")
    ves = load_csv("vessels_reference.csv")

    # ---- Optional: cargo/commodity for enrichment (safe to skip) ----
    try:
        cargo = load_csv("cargo_asof_2025-12-31.csv")
    except FileNotFoundError:
        cargo = pd.DataFrame(columns=["VOYAGE_ID", "CARGO_ID", "COMMODITY_ID"])
    try:
        cmdy = load_csv("commodity_reference.csv")
    except FileNotFoundError:
        cmdy = pd.DataFrame(columns=["COMMODITY_ID", "COMMODITY_GROUP_ID", "COMMODITY_CODE"])

    # ---- Select minimal columns we need from each table ----
    # Port calls: label + keys
    pc_cols = [
        "PORT_CALL_ID",
        "VOYAGE_ID",
        "PORT_ID",
        "TERMINAL_ID",
        "IS_BALLAST",
        "COMMODITY_ID",
        "DAYS_IN_PORT",
        "DAYS_STOPPAGES",
        "DAYS_EXTRA_IN_PORT",
    ]
    for c in pc_cols:
        if c not in pc.columns:
            pc[c] = np.nan
    pc = pc[pc_cols].copy()

    # Coerce numerics; filter clearly bad durations (keep ~1h..10d)
    num_cols_pc = [
        "PORT_CALL_ID",
        "VOYAGE_ID",
        "PORT_ID",
        "TERMINAL_ID",
        "IS_BALLAST",
        "COMMODITY_ID",
        "DAYS_IN_PORT",
        "DAYS_STOPPAGES",
        "DAYS_EXTRA_IN_PORT",
    ]
    for c in num_cols_pc:
        pc[c] = pd.to_numeric(pc[c], errors="coerce")
    pc = pc[pc["DAYS_IN_PORT"].between(0.04, 10.0)]

    # Voyages: vessel link + start date for season
    voy_cols = ["VOYAGE_ID", "VESSEL_ID", "VESSEL_TYPE_ID", "ESTIMATED_VOYAGE_START_DATE", "HAS_CANAL_PASSAGE"]
    for c in voy_cols:
        if c not in voy.columns:
            voy[c] = np.nan
    voy = voy[voy_cols].copy()
    voy["VOYAGE_ID"] = pd.to_numeric(voy["VOYAGE_ID"], errors="coerce")
    voy["VESSEL_ID"] = pd.to_numeric(voy["VESSEL_ID"], errors="coerce")
    voy["VESSEL_TYPE_ID"] = pd.to_numeric(voy["VESSEL_TYPE_ID"], errors="coerce")
    voy["HAS_CANAL_PASSAGE"] = pd.to_numeric(voy["HAS_CANAL_PASSAGE"], errors="coerce")
    voy["ESTIMATED_VOYAGE_START_DATE"] = pd.to_datetime(voy["ESTIMATED_VOYAGE_START_DATE"], errors="coerce")
    voy["MONTH_NO"] = voy["ESTIMATED_VOYAGE_START_DATE"].dt.month

    # Vessels: size proxies
    ves_cols = ["VESSEL_ID", "VESSEL_TYPE_ID", "DWT_SUMMER", "DRAFT_SUMMER", "LOA", "BEAM"]
    for c in ves_cols:
        if c not in ves.columns:
            ves[c] = np.nan
    ves = ves[ves_cols].copy()
    for c in ["VESSEL_ID", "VESSEL_TYPE_ID", "DWT_SUMMER", "DRAFT_SUMMER", "LOA", "BEAM"]:
        ves[c] = pd.to_numeric(ves[c], errors="coerce")

    # Optional cargo → commodity group enrichment (coarse proxy for handling)
    cg_cols = ["VOYAGE_ID", "COMMODITY_ID"]
    cargo = cargo[[c for c in cg_cols if c in cargo.columns]].dropna(
        subset=[c for c in cg_cols if c in cargo.columns], how="all"
    )
    for c in cargo.columns:
        cargo[c] = pd.to_numeric(cargo[c], errors="coerce")

    cmdy_cols = ["COMMODITY_ID", "COMMODITY_GROUP_ID", "COMMODITY_CODE"]
    cmdy = cmdy[[c for c in cmdy_cols if c in cmdy.columns]]
    if "COMMODITY_ID" in cmdy.columns:
        cmdy["COMMODITY_ID"] = pd.to_numeric(cmdy["COMMODITY_ID"], errors="coerce")

    # ---- Join: PortCall -> Voyage (vessel & season) ----
    df = pc.merge(
        voy[["VOYAGE_ID", "VESSEL_ID", "VESSEL_TYPE_ID", "MONTH_NO", "HAS_CANAL_PASSAGE"]], on="VOYAGE_ID", how="left"
    )

    # If VESSEL_TYPE_ID missing via voyage, fill from vessel table
    df = df.merge(
        ves[["VESSEL_ID", "VESSEL_TYPE_ID", "DWT_SUMMER", "DRAFT_SUMMER", "LOA", "BEAM"]].rename(
            columns={"VESSEL_TYPE_ID": "VESSEL_TYPE_ID_VE"}
        ),
        on="VESSEL_ID",
        how="left",
    )
    df["VESSEL_TYPE_ID"] = df["VESSEL_TYPE_ID"].fillna(df["VESSEL_TYPE_ID_VE"])
    df.drop(columns=["VESSEL_TYPE_ID_VE"], inplace=True)

    # Optional: attach commodity group via cargo->commodity
    if not cargo.empty and "COMMODITY_ID" in df.columns:
        # Pick one cargo per voyage (if multiple, just keep the first commodity as a coarse proxy)
        cg = cargo.dropna(subset=["VOYAGE_ID"]).sort_values("VOYAGE_ID").drop_duplicates("VOYAGE_ID")
        df = df.merge(cg, on="VOYAGE_ID", how="left", suffixes=("", "_FROM_CARGO"))
        # Prefer port-call commodity if present; otherwise take from cargo
        if "COMMODITY_ID_FROM_CARGO" in df.columns:
            df["COMMODITY_ID"] = df["COMMODITY_ID"].fillna(df["COMMODITY_ID_FROM_CARGO"])
            df.drop(columns=["COMMODITY_ID_FROM_CARGO"], inplace=True)
        # Commodity lookup
        if not cmdy.empty:
            df = df.merge(cmdy, on="COMMODITY_ID", how="left")

    # ---- Light outlier trimming within coarse groups (protect medians later) ----
    # Trim inside (PORT_ID, TERMINAL_ID, IS_BALLAST) groups to remove 5% tails.
    group_keys = [k for k in ["PORT_ID", "TERMINAL_ID", "IS_BALLAST"] if k in df.columns]
    if group_keys:
        grouped = df.groupby(group_keys, dropna=False)["DAYS_IN_PORT"]
        q05 = grouped.transform(lambda s: s.quantile(0.05))
        q95 = grouped.transform(lambda s: s.quantile(0.95))
        mask = (df["DAYS_IN_PORT"] >= q05) & (df["DAYS_IN_PORT"] <= q95)
        mask |= q05.isna() | q95.isna()
        df = df[mask]

    # ---- Final column set for the training dataset ----
    keep = [
        "PORT_CALL_ID",
        "VOYAGE_ID",
        "PORT_ID",
        "TERMINAL_ID",
        "IS_BALLAST",
        "VESSEL_ID",
        "VESSEL_TYPE_ID",
        "MONTH_NO",
        "HAS_CANAL_PASSAGE",
        "DWT_SUMMER",
        "DRAFT_SUMMER",
        "LOA",
        "BEAM",
        # label
        "DAYS_IN_PORT",
        # optional enrichment
        "COMMODITY_ID",
        "COMMODITY_GROUP_ID",
        "COMMODITY_CODE",
        "DAYS_STOPPAGES",
        "DAYS_EXTRA_IN_PORT",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep].copy()

    # ---- Write outputs ----
    out_csv = DER / "port_turnaround_training.csv"
    out_par = DER / "port_turnaround_training.parquet"
    df.to_csv(out_csv, index=False)
    try_write_parquet(df, out_par)

    # QA: counts per coarse grouping to see where you have support
    qa = (
        df.groupby(["PORT_ID", "TERMINAL_ID", "IS_BALLAST"], dropna=False)
        .size()
        .reset_index(name="n_obs")
        .sort_values("n_obs", ascending=False)
    )
    qa_path = DER / "port_turnaround_training_counts.csv"
    qa.to_csv(qa_path, index=False)

    rel_out_csv = out_csv.relative_to(ROOT)
    rel_qa = qa_path.relative_to(ROOT)
    print(f"[{paths.key}] wrote {len(df):,} rows -> {rel_out_csv}")
    if out_par.exists():
        rel_out_par = out_par.relative_to(ROOT)
        print(f"[{paths.key}] parquet -> {rel_out_par}")
    print(f"[{paths.key}] QA counts -> {rel_qa} (check for very small groups)")


if __name__ == "__main__":
    main()
