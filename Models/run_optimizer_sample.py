from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Models.optimizer import ColumnHints, ModelAdapters, OptimizerRequest, ScenarioWindow  # type: ignore
from Models.optimizer_mip import solve_mip  # type: ignore


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


def main() -> None:
    base = Path("DataSets/Derived/Stena/SampleScenario")
    vessels = pd.read_csv(base / "vessels.csv")
    voyages = pd.read_csv(base / "unallocated_voyages.csv")
    fleet_plan = pd.read_csv(base / "fleet_plan_voyages.csv")

    # Derive laycan columns if missing: use voyage start date as a proxy window.
    if "LAYCAN_START_UTC" not in voyages.columns:
        voyages["LAYCAN_START_UTC"] = voyages["ESTIMATED_VOYAGE_START_DATE"]
    if "LAYCAN_END_UTC" not in voyages.columns:
        voyages["LAYCAN_END_UTC"] = voyages["ESTIMATED_VOYAGE_START_DATE"]

    # Parse date columns to avoid per-row parsing warnings.
    date_cols = ["ESTIMATED_VOYAGE_START_DATE", "LAYCAN_START_UTC", "LAYCAN_END_UTC"]
    for col in date_cols:
        if col in voyages.columns:
            voyages[col] = pd.to_datetime(voyages[col], errors="coerce", utc=True)

    if "VESSEL_AVAILABLE_UTC" in vessels.columns:
        vessels["VESSEL_AVAILABLE_UTC"] = pd.to_datetime(vessels["VESSEL_AVAILABLE_UTC"], errors="coerce", utc=True)

    # Use the first fleet plan id present (if any)
    fleet_plan_key = str(fleet_plan["FLEET_PLAN_ID"].iloc[0]) if not fleet_plan.empty else "unknown"

    adapters = ModelAdapters(
        turnaround=ProxyTurnaroundPredictor(),  # replace with real turnaround adapter
        sailing_time=ProxySailingPredictor(),  # replace with real sailing-time adapter
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
            "name": "baseline_proxy",
            "max_arcs_per_job": 5,
            "unserved_penalty": 1000.0,
            "time_limit_sec": 30,
            "vessel_limit": None,
            "voyage_limit": None,
        },
        {
            "name": "tuned_low_penalty",
            "max_arcs_per_job": 10,
            "unserved_penalty": 100.0,
            "time_limit_sec": 60,
            "vessel_limit": None,
            "voyage_limit": None,
        },
        {
            "name": "subset_test",
            "max_arcs_per_job": 10,
            "unserved_penalty": 100.0,
            "time_limit_sec": 60,
            "vessel_limit": 10,
            "voyage_limit": 30,
        },
    ]

    for exp in experiments:
        vv = vessels.copy()
        vg = voyages.copy()
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
            time_limit_sec=exp["time_limit_sec"],
            window_slack_days=2.0,
            unserved_penalty=exp["unserved_penalty"],
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


if __name__ == "__main__":
    main()
