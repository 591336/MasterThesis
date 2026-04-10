from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import json

from Models.optimizer import ColumnHints, ModelAdapters, OptimizerRequest, ScenarioWindow  # type: ignore
from Models.optimizer_mip import solve_mip  # type: ignore
from Models.optimizer_adapters import SailingTimeModelAdapter, TurnaroundModelAdapter  # type: ignore


@dataclass
class DummyPredictor:
    """Simple adapter that returns zeros and exposes no required features."""

    feature_columns: tuple[str, ...] = ()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df), dtype=float)

    def fallback(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df), dtype=float)


@dataclass
class ProxySailingPredictor:
    """Heuristic sailing-time predictor using MILES_DIRECT and an assumed speed."""

    speed_knots: float = 12.0
    feature_columns: tuple[str, ...] = ("MILES_DIRECT",)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        miles = pd.to_numeric(df.get("MILES_DIRECT"), errors="coerce")
        hours = miles / self.speed_knots
        days = hours / 24.0
        days = days.fillna(1.0).to_numpy(dtype=float)
        return days

    def fallback(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), 1.0, dtype=float)


@dataclass
class ProxyTurnaroundPredictor:
    """Heuristic turnaround predictor: constant default days."""

    default_days: float = 1.0
    feature_columns: tuple[str, ...] = ()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.default_days, dtype=float)

    def fallback(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.default_days, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optimizer MIP experiments on the Stena sample scenario.")
    parser.add_argument(
        "--experiment",
        action="append",
        help="Experiment name to run (can be repeated). If omitted, runs all.",
    )
    parser.add_argument(
        "--time-limit-sec",
        type=int,
        help="Optional override of time_limit_sec for all experiments.",
    )
    parser.add_argument(
        "--laycan-end",
        choices=("hard", "soft"),
        help="Override laycan-end policy for all experiments (default: use experiment config).",
    )
    parser.add_argument(
        "--max-late-days",
        type=float,
        help="If using soft laycan end, cap allowed lateness (days).",
    )
    parser.add_argument(
        "--late-penalty-scale",
        type=float,
        help="If using soft laycan end, set penalty scale relative to unserved_penalty (default 0.001).",
    )
    parser.add_argument(
        "--missing-reposition-days",
        type=float,
        default=1.0,
        help="Fallback reposition time (days) when pairwise port distance is unavailable.",
    )
    parser.add_argument(
        "--enable-nonoverlap",
        action="store_true",
        help="Enable expensive disjunctive non-overlap constraints for manual experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path("DataSets/Derived/Stena/SampleScenario")
    vessels = pd.read_csv(base / "vessels_with_availability.csv")
    voyages = pd.read_csv(base / "unallocated_voyages_detailed.csv")
    fleet_plan = pd.read_csv(base / "fleet_plan_voyages.csv")
    manual_plan = pd.read_csv(base / "fleet_plan_manual.csv") if (base / "fleet_plan_manual.csv").exists() else None
    ports_latlon_path = Path("DataSets/Derived/Stena/Static/ports_latlon.csv")
    ports_latlon = pd.read_csv(ports_latlon_path) if ports_latlon_path.exists() else pd.DataFrame()

    # Derive laycan columns if missing: use voyage start date as a proxy window.
    if "LAYCAN_START_UTC" not in voyages.columns:
        voyages["LAYCAN_START_UTC"] = voyages["ESTIMATED_VOYAGE_START_DATE"]
    if "LAYCAN_END_UTC" not in voyages.columns:
        voyages["LAYCAN_END_UTC"] = voyages["ESTIMATED_VOYAGE_START_DATE"]

    # Parse date columns with explicit format to avoid per-row parsing warnings.
    date_cols = ["ESTIMATED_VOYAGE_START_DATE", "LAYCAN_START_UTC", "LAYCAN_END_UTC"]
    for col in date_cols:
        if col in voyages.columns:
            voyages[col] = pd.to_datetime(voyages[col], errors="coerce", utc=True, format="%d-%b-%y")

    if "VESSEL_AVAILABLE_UTC" in vessels.columns:
        vessels["VESSEL_AVAILABLE_UTC"] = pd.to_datetime(
            vessels["VESSEL_AVAILABLE_UTC"], errors="coerce", utc=True, format="%d-%b-%y"
        )

    # Add haversine distance (nm) as a fallback distance if we have port coordinates.
    if not ports_latlon.empty:
        def haversine_nm(lat1, lon1, lat2, lon2) -> float:
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            # Earth radius in nautical miles
            return 3440.065 * c

        port_map = ports_latlon.set_index("PORT_ID")[["LATITUDE", "LONGITUDE"]]
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
        voyages["MILES_DIRECT"] = voyages["MILES_DIRECT"].fillna(voyages["HV_DISTANCE_NM"])

    # Use the first fleet plan id present (if any)
    fleet_plan_key = str(fleet_plan["FLEET_PLAN_ID"].iloc[0]) if not fleet_plan.empty else "unknown"

    # Load sailing-time model and metadata; load turnaround model metadata for future use.
    sailing_meta = json.loads(Path("DataSets/Derived/Stena/ML/sailing_time_features.json").read_text())
    sailing_artifact = Path("Models/Artifacts/stena/sailing_time_dt.joblib")
    turnaround_meta = json.loads(Path("DataSets/Derived/Stena/ML/port_turnaround_features.json").read_text())
    turnaround_artifact = Path("Models/Artifacts/stena/port_turnaround_dt.joblib")

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

    experiments = [
        {
            "name": "subset_ml_proxy",
            "max_arcs_per_job": 15,
            "unserved_penalty": 5_000_000.0,  # stronger than median profit to encourage serving more jobs
            "time_limit_sec": 60,
            "vessel_limit": 10,
            "voyage_limit": 30,
        }
        ,
        {
            # More "physical": hard laycan end means infeasible voyages remain unserved rather than scheduled late.
            "name": "subset_physical",
            "max_arcs_per_job": 15,
            "unserved_penalty": 1_000_000.0,
            "time_limit_sec": 60,
            "vessel_limit": 10,
            "voyage_limit": 30,
            "hard_laycan_end": True,
        },
        {
            "name": "manual_match",
            "max_arcs_per_job": 6,
            "unserved_penalty": 5_000_000.0,
            "time_limit_sec": 90,
            "vessel_limit": None,
            "voyage_limit": None,  # filter to manual voyages if available
            "job_job_selection": "reposition",
            "job_job_candidate_window": 30,
        },
        {
            # Manual set with strict laycans: should only serve voyages that can start within window.
            "name": "manual_match_physical",
            "max_arcs_per_job": 6,
            "unserved_penalty": 10_000_000.0,
            "time_limit_sec": 90,
            "vessel_limit": None,
            "voyage_limit": None,
            "hard_laycan_end": True,
            "job_job_selection": "reposition",
            "job_job_candidate_window": 30,
        },
    ]

    selected = set(args.experiment) if args.experiment else None
    for exp in experiments:
        if selected is not None and exp["name"] not in selected:
            continue

        vv = vessels.copy()
        vg = voyages.copy()
        if exp["name"] in {"manual_match", "manual_match_physical"} and manual_plan is not None:
            manual_ids = set(manual_plan["VOYAGE_ID"].astype(str))
            vg = vg[vg["VOYAGE_ID"].astype(str).isin(manual_ids)]
            vessel_ids = set(manual_plan["VESSEL_ID"].astype(str))
            vv = vv[vv["VESSEL_ID"].astype(str).isin(vessel_ids)]
        if exp["vessel_limit"]:
            vv = vv.head(exp["vessel_limit"])
        if exp["voyage_limit"]:
            vg = vg.head(exp["voyage_limit"])

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
            max_arcs_per_job=exp["max_arcs_per_job"],
            time_limit_sec=int(args.time_limit_sec) if args.time_limit_sec is not None else exp["time_limit_sec"],
            window_slack_days=3.0,
            unserved_penalty=exp["unserved_penalty"],
            enforce_nonoverlap=(args.enable_nonoverlap and exp["name"] in {"manual_match", "manual_match_physical"}),
            hard_laycan_end=(
                (args.laycan_end == "hard")
                if args.laycan_end is not None
                else bool(exp.get("hard_laycan_end", False))
            ),
            max_late_days=args.max_late_days,
            late_penalty_scale=float(args.late_penalty_scale) if args.late_penalty_scale is not None else 0.001,
            job_job_selection=str(exp.get("job_job_selection", "time")),
            job_job_candidate_window=exp.get("job_job_candidate_window"),
            preferred_vessels=(
                dict(zip(manual_plan["VOYAGE_ID"].astype(str), manual_plan["VESSEL_ID"].astype(str)))
                if (manual_plan is not None and exp["name"] in {"manual_match", "manual_match_physical"})
                else None
            ),
            switch_penalty=(0.5 if exp["name"] in {"manual_match", "manual_match_physical"} else 0.0),
            time_weight=0.1,
            missing_reposition_days=float(args.missing_reposition_days),
        )

        print(f"\n=== Experiment: {exp['name']} ===")
        print(f"Vessels: {len(vv)}, Voyages: {len(vg)}")
        print("Logs:")
        for line in result.logs:
            print(f"  - {line}")
        print(f"Actions ({len(result.actions)}):")
        for action in result.actions[:5]:
            print(f"  {action}")
        if len(result.actions) > 5:
            print(f"  ... {len(result.actions) - 5} more")
        # Save actions for downstream comparison/visuals
        out_path = Path("Visualizations/output") / f"allocations_{exp['name']}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if result.actions:
            pd.DataFrame(
                [
                    {
                        "fleet_plan_key": a.fleet_plan_key,
                        "vessel_key": a.vessel_key,
                        "voyage_key": a.voyage_key,
                        "sequence": a.sequence,
                    }
                    for a in result.actions
                ]
            ).to_csv(out_path, index=False)
            print(f"Saved allocations to {out_path}")
        else:
            print("No actions to save.")

        if result.schedule is not None and not result.schedule.empty:
            schedule_path = Path("Visualizations/output") / f"schedule_{exp['name']}.csv"
            result.schedule.to_csv(schedule_path, index=False)
            print(f"Saved schedule to {schedule_path}")


if __name__ == "__main__":
    main()
