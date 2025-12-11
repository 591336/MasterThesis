from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence

import pandas as pd
import numpy as np


class TurnaroundPredictor(Protocol):
    """Adapter interface for port turnaround predictions."""

    feature_columns: Sequence[str]

    def predict(self, df: pd.DataFrame) -> np.ndarray: ...

    def fallback(self, df: pd.DataFrame) -> np.ndarray: ...


class SailingTimePredictor(Protocol):
    """Adapter interface for sailing-time predictions."""

    feature_columns: Sequence[str]

    def predict(self, df: pd.DataFrame) -> np.ndarray: ...

    def fallback(self, df: pd.DataFrame) -> np.ndarray: ...


@dataclass
class ModelAdapters:
    turnaround: TurnaroundPredictor
    sailing_time: SailingTimePredictor


@dataclass
class ColumnHints:
    """Column names expected in incoming DataFrames (kept configurable)."""

    vessel_key: str = "VESSEL_ID"
    voyage_key: str = "VOYAGE_ID"
    start_date: Optional[str] = "VOYAGE_START_DATE"
    laycan_start: Optional[str] = None
    laycan_end: Optional[str] = None
    vessel_available: Optional[str] = None


@dataclass
class ScenarioWindow:
    start_date: Optional[pd.Timestamp]
    end_date: Optional[pd.Timestamp]


@dataclass
class OptimizerRequest:
    scenario_key: str
    scenario_code: str
    fleet_plan_key: str
    is_budget: bool
    date_window: ScenarioWindow
    vessels: pd.DataFrame
    fleet_plan_voyages: pd.DataFrame
    unallocated_voyages: pd.DataFrame
    open_positions: pd.DataFrame
    cargos: Optional[pd.DataFrame] = None
    compare_fleet_plan_voyages: Optional[pd.DataFrame] = None
    seed: int = 42
    column_hints: ColumnHints = field(default_factory=ColumnHints)


@dataclass
class OptimizerAction:
    action_type: str


@dataclass
class AllocateVoyage(OptimizerAction):
    fleet_plan_key: str
    vessel_key: str
    voyage_key: str
    sequence: int
    action_type: str = field(init=False, default="allocate")


@dataclass
class MoveVoyage(OptimizerAction):
    fleet_plan_voyage_key: str
    vessel_key: str
    voyage_key: str
    sequence: int
    action_type: str = field(init=False, default="move")


@dataclass
class UnallocateVoyage(OptimizerAction):
    fleet_plan_voyage_key: str
    action_type: str = field(init=False, default="unallocate")


@dataclass
class OptimizerResult:
    actions: List[OptimizerAction]
    logs: List[str]


class GreedyOptimizer:
    """Minimal scaffold: assigns unallocated voyages in order to available vessels."""

    def __init__(self, models: ModelAdapters):
        self.models = models

    def plan(self, request: OptimizerRequest) -> OptimizerResult:
        rng = random.Random(request.seed)
        logs: List[str] = []
        actions: List[OptimizerAction] = []

        if request.is_budget:
            logs.append("Skipping budget scenario; optimizer only runs on operational scenarios.")
            return OptimizerResult(actions=actions, logs=logs)

        if request.vessels.empty:
            logs.append("No vessels available for allocation.")
            return OptimizerResult(actions=actions, logs=logs)

        if request.unallocated_voyages.empty:
            logs.append("No unallocated voyages to schedule.")
            return OptimizerResult(actions=actions, logs=logs)

        col_vessel = request.column_hints.vessel_key
        col_voyage = request.column_hints.voyage_key
        col_start = request.column_hints.start_date

        if col_vessel not in request.vessels.columns or col_voyage not in request.unallocated_voyages.columns:
            logs.append(f"Missing expected columns ({col_vessel}, {col_voyage}); no actions produced.")
            return OptimizerResult(actions=actions, logs=logs)

        vessels = request.vessels.copy()
        voyages = request.unallocated_voyages.copy()

        if col_start and col_start in voyages.columns:
            voyages[col_start] = pd.to_datetime(voyages[col_start], errors="coerce")
            voyages = voyages.sort_values(col_start, na_position="last")
        else:
            voyages = voyages.sample(frac=1.0, random_state=rng.randint(0, 10_000))
            logs.append("Voyages shuffled (no start-date column provided).")

        vessel_keys = vessels[col_vessel].dropna().astype(str).tolist()
        if not vessel_keys:
            logs.append("No vessel keys available after filtering.")
            return OptimizerResult(actions=actions, logs=logs)

        for idx, (_, voyage) in enumerate(voyages.iterrows()):
            vessel_key = vessel_keys[idx % len(vessel_keys)]
            voyage_key = str(voyage[col_voyage])
            sequence = idx + 1
            actions.append(
                AllocateVoyage(
                    fleet_plan_key=request.fleet_plan_key,
                    vessel_key=vessel_key,
                    voyage_key=voyage_key,
                    sequence=sequence,
                    action_type="allocate",
                )
            )

        logs.append(
            f"Planned {len(actions)} allocate actions across {len(vessel_keys)} vessels "
            f"(seed={request.seed})."
        )
        logs.append("NOTE: Greedy placeholder ignores constraints and model scores; replace with search logic.")

        return OptimizerResult(actions=actions, logs=logs)
