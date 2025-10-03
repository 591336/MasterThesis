from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "DataSets" / "Raw" / "NorthernLightsTest"
DER = ROOT / "DataSets" / "Derived"
DER.mkdir(parents=True, exist_ok=True)


def load_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / name)
    df.columns = [c.upper() for c in df.columns]
    return df


def try_write_parquet(df: pd.DataFrame, path: Path) -> bool:
    """Write parquet if an engine is installed, otherwise emit a gentle reminder."""
    try:
        df.to_parquet(path, index=False)
        return True
    except ImportError:
        print(
            f"Skipping parquet output ({path.name}): install 'pyarrow' or 'fastparquet' to enable parquet exports."
        )
        return False


# ---- Load raw tables ----
pc = load_csv("port_calls_completed_asof_2025-12-31.csv")
voy = load_csv("voyages_completed_asof_2025-12-31.csv")
ves = load_csv("vessels_reference.csv")
# Optional enrichments (safe if missing)
try:
    cargo = load_csv("cargo_asof_2025-12-31.csv")
except FileNotFoundError:
    cargo = pd.DataFrame(columns=["VOYAGE_ID", "CARGO_ID", "COMMODITY_ID"])
try:
    cmdy = load_csv("commodity_reference.csv")
except FileNotFoundError:
    cmdy = pd.DataFrame(columns=["COMMODITY_ID", "COMMODITY_GROUP_ID", "COMMODITY_CODE"])

# ---- Select / coerce minimal columns we need ----
# Port calls
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
# Numeric coercions
for c in [
    "PORT_CALL_ID",
    "VOYAGE_ID",
    "PORT_ID",
    "TERMINAL_ID",
    "IS_BALLAST",
    "COMMODITY_ID",
    "DAYS_IN_PORT",
    "DAYS_STOPPAGES",
    "DAYS_EXTRA_IN_PORT",
]:
    pc[c] = pd.to_numeric(pc[c], errors="coerce")

# Basic sanity: positive-ish durations, drop absurd values
pc = pc[pc["DAYS_IN_PORT"].between(0.04, 10.0)]  # ~1h..10d

# Voyages (to fetch vessel id/type)
voy_cols = ["VOYAGE_ID", "VESSEL_ID", "VESSEL_TYPE_ID", "ESTIMATED_VOYAGE_START_DATE"]
for c in voy_cols:
    if c not in voy.columns:
        voy[c] = np.nan
voy = voy[voy_cols].copy()
voy["VOYAGE_ID"] = pd.to_numeric(voy["VOYAGE_ID"], errors="coerce")
voy["VESSEL_ID"] = pd.to_numeric(voy["VESSEL_ID"], errors="coerce")
voy["VESSEL_TYPE_ID"] = pd.to_numeric(voy["VESSEL_TYPE_ID"], errors="coerce")
voy["ESTIMATED_VOYAGE_START_DATE"] = pd.to_datetime(voy["ESTIMATED_VOYAGE_START_DATE"], errors="coerce")
voy["MONTH_NO"] = voy["ESTIMATED_VOYAGE_START_DATE"].dt.month

# Vessels (backup for type in case voyages lacks it)
ves_cols = ["VESSEL_ID", "VESSEL_TYPE_ID", "DWT_SUMMER", "DRAFT_SUMMER"]
for c in ves_cols:
    if c not in ves.columns:
        ves[c] = np.nan
ves = ves[ves_cols].copy()
ves["VESSEL_ID"] = pd.to_numeric(ves["VESSEL_ID"], errors="coerce")
ves["VESSEL_TYPE_ID"] = pd.to_numeric(ves["VESSEL_TYPE_ID"], errors="coerce")

# Optional cargo/commodity (for enrichment / grouping later)
cargo_cols = ["VOYAGE_ID", "CARGO_ID", "COMMODITY_ID"]
cargo = cargo[[c for c in cargo_cols if c in cargo.columns]].copy()
for c in cargo.columns:
    cargo[c] = pd.to_numeric(cargo[c], errors="coerce")

cmdy_cols = ["COMMODITY_ID", "COMMODITY_GROUP_ID", "COMMODITY_CODE"]
cmdy = cmdy[[c for c in cmdy_cols if c in cmdy.columns]].copy()
for c in cmdy.columns:
    cmdy[c] = pd.to_numeric(cmdy[c], errors="coerce") if c.endswith("_ID") else cmdy[c]

# ---- Join keys: port_call -> voyage -> vessel type / month ----
pc = pc.merge(voy[["VOYAGE_ID", "VESSEL_ID", "VESSEL_TYPE_ID", "MONTH_NO"]], on="VOYAGE_ID", how="left")
# If vessel_type_id missing via voyage, fill from vessel table
pc = pc.merge(
    ves[["VESSEL_ID", "VESSEL_TYPE_ID"]].rename(columns={"VESSEL_TYPE_ID": "VESSEL_TYPE_ID_VE"}),
    on="VESSEL_ID",
    how="left",
)
pc["VESSEL_TYPE_ID"] = pc["VESSEL_TYPE_ID"].fillna(pc["VESSEL_TYPE_ID_VE"])
pc.drop(columns=["VESSEL_TYPE_ID_VE"], inplace=True)

# Optional: attach commodity group (coarse proxy)
if "COMMODITY_ID" in pc.columns and not pc["COMMODITY_ID"].isna().all() and not cmdy.empty:
    pc = pc.merge(cmdy, on="COMMODITY_ID", how="left")

# ---- Robust trimming within groups to remove extreme durations ----
# We'll trim within coarse groups to avoid median distortion:
trim_keys = ["PORT_ID", "TERMINAL_ID", "IS_BALLAST"]


def trim_group(g: pd.DataFrame) -> pd.DataFrame:
    q05 = g["DAYS_IN_PORT"].quantile(0.05)
    q95 = g["DAYS_IN_PORT"].quantile(0.95)
    return g[(g["DAYS_IN_PORT"] >= q05) & (g["DAYS_IN_PORT"] <= q95)]


group_cols = [k for k in trim_keys if k in pc.columns]
pc_trim = pc.groupby(group_cols, dropna=False, group_keys=False).apply(
    trim_group, include_groups=False
)
missing_cols = [col for col in group_cols if col not in pc_trim.columns]
if missing_cols:
    pc_trim = pc_trim.join(pc[missing_cols], how="left")

# ---- Build fallback hierarchy ----
# Most specific → least specific
# 1) PORT, TERMINAL, IS_BALLAST, VESSEL_TYPE_ID, MONTH_NO
# 2) PORT, TERMINAL, IS_BALLAST, VESSEL_TYPE_ID
# 3) PORT, TERMINAL, IS_BALLAST
# 4) PORT, IS_BALLAST
# 5) PORT
# 6) GLOBAL (single median) – for safety only

hierarchy = [
    ["PORT_ID", "TERMINAL_ID", "IS_BALLAST", "VESSEL_TYPE_ID", "MONTH_NO"],
    ["PORT_ID", "TERMINAL_ID", "IS_BALLAST", "VESSEL_TYPE_ID"],
    ["PORT_ID", "TERMINAL_ID", "IS_BALLAST"],
    ["PORT_ID", "IS_BALLAST"],
    ["PORT_ID"],
    [],  # global
]


def agg_by(keys):
    if not keys:  # global
        stats = {
            "median_days_in_port": pc_trim["DAYS_IN_PORT"].median(),
            "p10": pc_trim["DAYS_IN_PORT"].quantile(0.10),
            "p90": pc_trim["DAYS_IN_PORT"].quantile(0.90),
            "n_obs": pc_trim["DAYS_IN_PORT"].size,
        }
        df = pd.DataFrame([stats])
        for c in ["PORT_ID", "TERMINAL_ID", "IS_BALLAST", "VESSEL_TYPE_ID", "MONTH_NO"]:
            df[c] = np.nan
        df = df[[
            "PORT_ID",
            "TERMINAL_ID",
            "IS_BALLAST",
            "VESSEL_TYPE_ID",
            "MONTH_NO",
            "median_days_in_port",
            "p10",
            "p90",
            "n_obs",
        ]]
        return df
    g = (
        pc_trim.groupby(keys, dropna=False)
        .agg(
            median_days_in_port=("DAYS_IN_PORT", "median"),
            p10=("DAYS_IN_PORT", lambda s: s.quantile(0.10)),
            p90=("DAYS_IN_PORT", lambda s: s.quantile(0.90)),
            n_obs=("DAYS_IN_PORT", "size"),
        )
        .reset_index()
    )
    return g


frames = []
for lvl, keys in enumerate(hierarchy, start=1):
    df = agg_by(keys)
    df["level"] = lvl
    frames.append(df)

lookup_all = pd.concat(frames, ignore_index=True)

# ---- Minimum support and guardrails ----
# Require at least 3 observations for any entry (except global which always exists)
lookup_all = lookup_all[(lookup_all["n_obs"] >= 3) | (lookup_all["level"] == len(hierarchy))]

# Clamp medians to sensible range (1h..10d)
lookup_all["median_days_in_port"] = lookup_all["median_days_in_port"].clip(lower=0.04, upper=10.0)


# Specificity score to help the solver pick the best row at query time
def spec(row):
    score = 0
    score += 8 if not pd.isna(row.get("PORT_ID")) else 0
    score += 4 if not pd.isna(row.get("TERMINAL_ID")) else 0
    score += 2 if not pd.isna(row.get("IS_BALLAST")) else 0
    score += 1 if not pd.isna(row.get("VESSEL_TYPE_ID")) else 0
    score += 1 if not pd.isna(row.get("MONTH_NO")) else 0
    return score


lookup_all["specificity_score"] = lookup_all.apply(spec, axis=1)

# Deduplicate: for identical key rows across levels, keep the most specific (highest score, then largest n_obs)
key_cols = ["PORT_ID", "TERMINAL_ID", "IS_BALLAST", "VESSEL_TYPE_ID", "MONTH_NO"]
lookup_all.sort_values(by=["specificity_score", "n_obs"], ascending=[False, False], inplace=True)
lookup = lookup_all.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)

# ---- Write outputs ----
lookup = lookup[key_cols + ["median_days_in_port", "n_obs", "specificity_score", "level"]]
try_write_parquet(lookup, DER / "port_turnaround_lookup.parquet")
lookup.to_csv(DER / "port_turnaround_lookup.csv", index=False)

# ---- QA snapshot ----
qa = (
    pc_trim.assign(BIN=lambda d: pd.cut(d["DAYS_IN_PORT"], bins=[0, 0.5, 1, 2, 3, 5, 10], include_lowest=True))
    .groupby("BIN", observed=False)
    .size()
    .reset_index(name="n")
)
qa.to_csv(DER / "port_turnaround_lookup_qa_bins.csv", index=False)

print(f"Wrote {len(lookup):,} rows -> {DER/'port_turnaround_lookup.csv'}")
print(f"QA bins -> {DER/'port_turnaround_lookup_qa_bins.csv'}")
