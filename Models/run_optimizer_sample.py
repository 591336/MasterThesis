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

    # Parse date columns with a format hint to avoid per-row parsing warnings.
    date_cols = ["ESTIMATED_VOYAGE_START_DATE", "LAYCAN_START_UTC", "LAYCAN_END_UTC"]
    for col in date_cols:
        if col in voyages.columns:
            voyages[col] = pd.to_datetime(voyages[col], errors="coerce", utc=True)

    if "VESSEL_AVAILABLE_UTC" in vessels.columns:
        vessels["VESSEL_AVAILABLE_UTC"] = pd.to_datetime(vessels["VESSEL_AVAILABLE_UTC"], errors="coerce", utc=True)

    # Use the first fleet plan id present (if any)
    fleet_plan_key = str(fleet_plan["FLEET_PLAN_ID"].iloc[0]) if not fleet_plan.empty else "unknown"

    adapters = ModelAdapters(
        turnaround=DummyPredictor(),  # replace with real turnaround adapter
        sailing_time=DummyPredictor(),  # replace with real sailing-time adapter
    )

    hints = ColumnHints(
        vessel_key="VESSEL_ID",
        voyage_key="VOYAGE_ID",
        start_date="ESTIMATED_VOYAGE_START_DATE",
        laycan_start="LAYCAN_START_UTC",
        laycan_end="LAYCAN_END_UTC",
        vessel_available="VESSEL_AVAILABLE_UTC",
    )

    request = OptimizerRequest(
        scenario_key="stena_sample",
        scenario_code="stena_sample",
        fleet_plan_key=fleet_plan_key,
        is_budget=False,
        date_window=ScenarioWindow(start_date=None, end_date=None),
        vessels=vessels,
        fleet_plan_voyages=fleet_plan,
        unallocated_voyages=voyages,
        open_positions=pd.DataFrame(),
        cargos=None,
        compare_fleet_plan_voyages=None,
        seed=42,
        column_hints=hints,
    )

    result = solve_mip(
        request,
        adapters,
        max_arcs_per_job=5,
        time_limit_sec=10,
        window_slack_days=2.0,
    )

    print("Logs:")
    for line in result.logs:
        print(f"  - {line}")

    print(f"\nActions ({len(result.actions)}):")
    for action in result.actions[:10]:
        print(f"  {action}")
    if len(result.actions) > 10:
        print(f"  ... {len(result.actions) - 10} more")


if __name__ == "__main__":
    main()
