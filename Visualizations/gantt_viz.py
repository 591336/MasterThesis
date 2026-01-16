"""
Generate a lightweight Gantt-style plot of the optimizer allocation for the sample scenario.

This keeps the optimizer logic untouched: we reuse the same data prep as `run_optimizer_sample`,
run the MIP on a small subset, and then plot per-vessel bars using voyage start dates and
durations (days at sea + in port). Output is written to Visualizations/output/.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import argparse
from typing import List

ROOT = Path(__file__).resolve().parents[1]
DATA_BASE = ROOT / "DataSets" / "Derived" / "Stena" / "SampleScenario"
STATIC_BASE = ROOT / "DataSets" / "Derived" / "Stena" / "Static"
OUTPUT_DIR = ROOT / "Visualizations" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Models.optimizer import ColumnHints, ModelAdapters, OptimizerRequest, ScenarioWindow  # type: ignore
from Models.optimizer_adapters import SailingTimeModelAdapter, TurnaroundModelAdapter  # type: ignore
from Models.optimizer_mip import solve_mip  # type: ignore


@dataclass
class GanttConfig:
    vessel_limit: int = 10
    voyage_limit: int = 30
    max_arcs_per_job: int = 15
    unserved_penalty: float = 5_000_000.0
    time_limit_sec: int = 60
    window_slack_days: float = 3.0


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vessels = pd.read_csv(DATA_BASE / "vessels_with_availability.csv")
    voyages = pd.read_csv(DATA_BASE / "unallocated_voyages_detailed.csv")
    fleet_plan = pd.read_csv(DATA_BASE / "fleet_plan_voyages.csv")
    ports_latlon_path = STATIC_BASE / "ports_latlon.csv"
    ports_latlon = pd.read_csv(ports_latlon_path) if ports_latlon_path.exists() else pd.DataFrame()
    return vessels, voyages, fleet_plan, ports_latlon


def prep_dates(vessels: pd.DataFrame, voyages: pd.DataFrame) -> None:
    if "LAYCAN_START_UTC" not in voyages.columns:
        voyages["LAYCAN_START_UTC"] = voyages["ESTIMATED_VOYAGE_START_DATE"]
    if "LAYCAN_END_UTC" not in voyages.columns:
        voyages["LAYCAN_END_UTC"] = voyages["ESTIMATED_VOYAGE_START_DATE"]

    date_cols = ["ESTIMATED_VOYAGE_START_DATE", "LAYCAN_START_UTC", "LAYCAN_END_UTC"]
    for col in date_cols:
        if col in voyages.columns:
            voyages[col] = pd.to_datetime(voyages[col], errors="coerce", utc=True, format="%d-%b-%y")

    if "VESSEL_AVAILABLE_UTC" in vessels.columns:
        vessels["VESSEL_AVAILABLE_UTC"] = pd.to_datetime(
            vessels["VESSEL_AVAILABLE_UTC"], errors="coerce", utc=True, format="%d-%b-%y"
        )


def add_haversine(voyages: pd.DataFrame, ports_latlon: pd.DataFrame) -> None:
    if ports_latlon.empty or "PORT_ID" not in ports_latlon.columns:
        return

    port_map = ports_latlon.set_index("PORT_ID")[["LATITUDE", "LONGITUDE"]]

    def haversine_nm(lat1, lon1, lat2, lon2) -> float:
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return 3440.065 * c  # Earth radius in nautical miles

    def calc_row(row):
        o = row.get("ORIGIN_PORT_ID")
        d = row.get("DEST_PORT_ID")
        if pd.isna(o) or pd.isna(d):
            return np.nan
        if o not in port_map.index or d not in port_map.index:
            return np.nan
        lat1, lon1 = port_map.loc[o]
        lat2, lon2 = port_map.loc[d]
        return haversine_nm(lat1, lon1, lat2, lon2)

    voyages["HV_DISTANCE_NM"] = voyages.apply(calc_row, axis=1)
    if "MILES_DIRECT" in voyages.columns:
        voyages["MILES_DIRECT"] = voyages["MILES_DIRECT"].fillna(voyages["HV_DISTANCE_NM"])
    else:
        voyages["MILES_DIRECT"] = voyages["HV_DISTANCE_NM"]


def run_optimizer(vessels: pd.DataFrame, voyages: pd.DataFrame, fleet_plan: pd.DataFrame, cfg: GanttConfig):
    sailing_meta = json.loads((ROOT / "DataSets" / "Derived" / "Stena" / "ML" / "sailing_time_features.json").read_text())
    sailing_artifact = ROOT / "Models" / "Artifacts" / "stena" / "sailing_time_dt.joblib"
    turnaround_meta = json.loads((ROOT / "DataSets" / "Derived" / "Stena" / "ML" / "port_turnaround_features.json").read_text())
    turnaround_artifact = ROOT / "Models" / "Artifacts" / "stena" / "port_turnaround_dt.joblib"

    adapters = ModelAdapters(
        turnaround=TurnaroundModelAdapter(turnaround_artifact, turnaround_meta),
        sailing_time=SailingTimeModelAdapter(sailing_artifact, sailing_meta),
    )

    hints = ColumnHints(
        vessel_key="VESSEL_ID",
        voyage_key="VOYAGE_ID",
        start_date="ESTIMATED_VOYAGE_START_DATE",
        laycan_start="LAYCAN_START_UTC",
        laycan_end="LAYCAN_END_UTC",
        vessel_available="VESSEL_AVAILABLE_UTC",
    )

    vv = vessels.head(cfg.vessel_limit).copy()
    vg = voyages.head(cfg.voyage_limit).copy()
    fleet_plan_key = str(fleet_plan["FLEET_PLAN_ID"].iloc[0]) if not fleet_plan.empty else "unknown"

    request = OptimizerRequest(
        scenario_key="stena_sample",
        scenario_code="stena_sample",
        fleet_plan_key=fleet_plan_key,
        is_budget=False,
        date_window=ScenarioWindow(start_date=None, end_date=None),
        vessels=vv,
        fleet_plan_voyages=fleet_plan,
        unallocated_voyages=vg,
        open_positions=pd.DataFrame(),
        cargos=None,
        compare_fleet_plan_voyages=None,
        seed=42,
        column_hints=hints,
    )

    result = solve_mip(
        request,
        adapters,
        max_arcs_per_job=cfg.max_arcs_per_job,
        time_limit_sec=cfg.time_limit_sec,
        window_slack_days=cfg.window_slack_days,
        unserved_penalty=cfg.unserved_penalty,
    )
    return result.actions, result.logs, result.schedule


def build_timeline(actions, vessels: pd.DataFrame, voyages: pd.DataFrame, schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    if schedule is not None and not schedule.empty:
        vessels = vessels.copy()
        vessels["VESSEL_ID"] = vessels["VESSEL_ID"].astype(str)
        vessel_names = vessels.set_index("VESSEL_ID").get("VESSEL_NAME", pd.Series(dtype="string"))

        timeline = schedule.copy()
        timeline["vessel_key"] = timeline["vessel_key"].astype(str)
        timeline["vessel_name"] = timeline["vessel_key"].map(vessel_names).fillna(timeline["vessel_key"])
        timeline.rename(columns={"vessel_key": "vessel_id", "voyage_key": "voyage_id"}, inplace=True)
        keep_cols = [
            "vessel_id",
            "vessel_name",
            "voyage_id",
            "sequence",
            "start",
            "end",
            "laycan_start",
            "laycan_end",
            "duration_days",
        ]
        for col in keep_cols:
            if col not in timeline.columns:
                timeline[col] = pd.NA
        return timeline[keep_cols].sort_values(["vessel_name", "sequence"]).reset_index(drop=True)

    if not actions:
        return pd.DataFrame()

    voyages = voyages.copy()
    voyages["VOYAGE_ID"] = voyages["VOYAGE_ID"].astype(str)
    vessels["VESSEL_ID"] = vessels["VESSEL_ID"].astype(str)

    voyages["DURATION_DAYS"] = (
        pd.to_numeric(voyages.get("DAYS_TOTAL_AT_SEA"), errors="coerce").fillna(0.0)
        + pd.to_numeric(voyages.get("DAYS_TOTAL_IN_PORT"), errors="coerce").fillna(0.0)
    )
    voyages["DURATION_DAYS"] = voyages["DURATION_DAYS"].replace(0, 1.0)  # fallback

    rows = []
    for action in actions:
        vrow = voyages.set_index("VOYAGE_ID").loc[str(action.voyage_key)]
        vessel_name = vessels.set_index("VESSEL_ID").get("VESSEL_NAME", pd.Series()).get(str(action.vessel_key), str(action.vessel_key))
        start_dt = vrow.get("ESTIMATED_VOYAGE_START_DATE")
        laycan_start = vrow.get("LAYCAN_START_UTC")
        laycan_end = vrow.get("LAYCAN_END_UTC")
        duration = vrow.get("DURATION_DAYS", 1.0)
        end_dt = None
        if isinstance(start_dt, pd.Timestamp):
            end_dt = start_dt + pd.to_timedelta(duration, unit="D")
        rows.append(
            {
                "vessel_id": str(action.vessel_key),
                "vessel_name": vessel_name,
                "voyage_id": str(action.voyage_key),
                "sequence": action.sequence,
                "start": start_dt,
                "end": end_dt,
                "laycan_start": laycan_start,
                "laycan_end": laycan_end,
                "duration_days": duration,
            }
        )
    df = pd.DataFrame(rows).sort_values(["vessel_name", "sequence"])
    return df


def plot_gantt(df: pd.DataFrame, title: str = "Optimizer Allocation – Sample Scenario") -> Path:
    if df.empty:
        raise ValueError("No data to plot")

    vessels = list(df["vessel_name"].unique())
    vessels.sort()
    vessel_to_y = {v: idx for idx, v in enumerate(vessels)}
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(vessels) + 2)))
    label_small_threshold = 45.0  # days; skip labels on very short bars to reduce overlap
    label_inside_threshold = 120.0  # days; if longer, center the label inside the bar

    # Limit x-range to data span with a small buffer to reduce skew
    start_times = [row["start"] for _, row in df.iterrows() if isinstance(row["start"], pd.Timestamp)]
    end_times = [row["end"] for _, row in df.iterrows() if isinstance(row["end"], pd.Timestamp)]
    if start_times and end_times:
        min_dt = min(start_times)
        max_dt = max(end_times)
        buffer = pd.Timedelta(days=30)
        ax.set_xlim(mdates.date2num(min_dt - buffer), mdates.date2num(max_dt + buffer))

    for _, row in df.iterrows():
        y = vessel_to_y[row["vessel_name"]]
        start = row["start"]
        end = row["end"]
        if not isinstance(start, pd.Timestamp) or not isinstance(end, pd.Timestamp):
            continue
        width = (end - start).total_seconds() / 86400.0
        color = cmap(y % 20)
        ax.barh(y, width, left=mdates.date2num(start), height=0.35, color=color, alpha=0.7)
        if width >= label_small_threshold:
            label = f"{row['voyage_id']} (seq {row['sequence']})"
            if width >= label_inside_threshold:
                x_pos = mdates.date2num(start + (end - start) / 2)
                ax.text(
                    x_pos,
                    y,
                    label,
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )
            else:
                ax.text(
                    mdates.date2num(start),
                    y + 0.25,
                    label,
                    fontsize=7,
                    color="black",
                )
        if isinstance(row.get("laycan_start"), pd.Timestamp) and isinstance(row.get("laycan_end"), pd.Timestamp):
            ax.hlines(y, mdates.date2num(row["laycan_start"]), mdates.date2num(row["laycan_end"]), colors="k", linestyles="--", linewidth=1, alpha=0.5)

    ax.set_yticks(list(vessel_to_y.values()))
    ax.set_yticklabels(vessels, fontsize=9)
    ax.xaxis_date()
    fig.autofmt_xdate()
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Vessel")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "gantt_sample.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(reuse_data: bool = False) -> None:
    cfg = GanttConfig()
    vessels, voyages, fleet_plan, ports_latlon = load_data()
    prep_dates(vessels, voyages)
    add_haversine(voyages, ports_latlon)

    timeline_path = OUTPUT_DIR / "gantt_schedule.csv"
    if reuse_data and timeline_path.exists():
        timeline = pd.read_csv(
            timeline_path,
            parse_dates=["start", "end", "laycan_start", "laycan_end"],
        )
        logs: List[str] = ["Reused cached schedule; optimizer not run."]
    else:
        actions, logs, schedule = run_optimizer(vessels, voyages, fleet_plan, cfg)
        timeline = build_timeline(actions, vessels, voyages, schedule=schedule)
        timeline.to_csv(timeline_path, index=False)
    out_path = plot_gantt(timeline)

    print("Logs:")
    for line in logs:
        print(f"  - {line}")
    print(f"Gantt written to: {out_path}")
    print(f"Schedule data written to: {timeline_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Gantt-style plot for optimizer output.")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Skip running optimizer and reuse cached gantt_schedule.csv if present.",
    )
    args = parser.parse_args()
    main(reuse_data=args.reuse)
