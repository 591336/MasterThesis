from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import pulp
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("pulp is required for optimizer_mip; install with `uv add pulp`.") from exc

from .optimizer import (
    AllocateVoyage,
    GreedyOptimizer,
    ModelAdapters,
    OptimizerAction,
    OptimizerRequest,
    OptimizerResult,
)


@dataclass
class ArcData:
    vessel: str
    i: str
    j: str
    sailing_time: float
    turnaround_time: float
    cost: float


def _predict_series(model, df: pd.DataFrame, fallback_value: float) -> np.ndarray:
    try:
        cols = [c for c in model.feature_columns if c in df.columns]
        features = df[cols].copy()
        return np.asarray(model.predict(features), dtype=float)
    except Exception:
        return np.full(len(df), fallback_value, dtype=float)


def _predict_row(model, row: pd.Series, fallback_value: float) -> float:
    try:
        cols = [c for c in model.feature_columns if c in row.index]
        df = pd.DataFrame([row[cols]])
        return float(np.asarray(model.predict(df), dtype=float)[0])
    except Exception:
        return fallback_value


def build_arcs(
    request: OptimizerRequest,
    models: ModelAdapters,
    max_arcs_per_job: int = 10,
    unserved_penalty: float = 1_000.0,
    window_slack_days: float = 2.0,
    default_speed_knots: float = 12.0,
    default_turnaround_days: float = 1.0,
) -> tuple[
    List[ArcData],
    Dict[str, float],
    Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]],
    Dict[str, Optional[pd.Timestamp]],
    float,
]:
    vessels = request.vessels.copy()
    voyages = request.unallocated_voyages.copy()

    col_vessel = request.column_hints.vessel_key
    col_voyage = request.column_hints.voyage_key
    col_start = request.column_hints.start_date
    col_laycan_start = request.column_hints.laycan_start
    col_laycan_end = request.column_hints.laycan_end
    col_vessel_avail = request.column_hints.vessel_available

    voyages[col_voyage] = voyages[col_voyage].astype(str)
    vessels[col_vessel] = vessels[col_vessel].astype(str)

    if col_start and col_start in voyages.columns:
        voyages[col_start] = pd.to_datetime(voyages[col_start], errors="coerce")
        voyages = voyages.sort_values(col_start, na_position="last")
    if col_laycan_start and col_laycan_start in voyages.columns:
        voyages[col_laycan_start] = pd.to_datetime(voyages[col_laycan_start], errors="coerce")
    if col_laycan_end and col_laycan_end in voyages.columns:
        voyages[col_laycan_end] = pd.to_datetime(voyages[col_laycan_end], errors="coerce")
    if col_vessel_avail and col_vessel_avail in vessels.columns:
        vessels[col_vessel_avail] = pd.to_datetime(vessels[col_vessel_avail], errors="coerce")

    voyage_keys = voyages[col_voyage].astype(str).tolist()
    vessel_keys = vessels[col_vessel].astype(str).tolist()
    if not voyage_keys or not vessel_keys:
        return [], {}, {}, {}, unserved_penalty

    voyage_rows = voyages.set_index(col_voyage)
    vessel_rows = vessels.set_index(col_vessel)

    turnaround_pred = _predict_series(models.turnaround, voyages, fallback_value=default_turnaround_days)
    turnaround_map = {vk: float(tt) for vk, tt in zip(voyage_keys, turnaround_pred)}

    time_windows: Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = {}
    for vk in voyage_keys:
        row = voyage_rows.loc[vk]
        a_j = row[col_laycan_start] if col_laycan_start and col_laycan_start in row else None
        b_j = row[col_laycan_end] if col_laycan_end and col_laycan_end in row else None
        if a_j is None and b_j is None and col_start and col_start in row:
            a_j = row[col_start]
            b_j = row[col_start]
        time_windows[str(vk)] = (a_j, b_j)

    vessel_available: Dict[str, Optional[pd.Timestamp]] = {}
    for vk in vessel_keys:
        avail = None
        if col_vessel_avail and col_vessel_avail in vessel_rows.columns:
            avail = vessel_rows.loc[vk][col_vessel_avail]
        vessel_available[vk] = avail if isinstance(avail, pd.Timestamp) else None

    def to_days(ts: Optional[pd.Timestamp], ref: pd.Timestamp) -> float:
        if not isinstance(ts, pd.Timestamp):
            return 0.0
        return float((ts - ref).total_seconds() / 86400.0)

    ref_candidates: List[pd.Timestamp] = []
    if isinstance(request.date_window.start_date, pd.Timestamp):
        ref_candidates.append(request.date_window.start_date)
    for tw in time_windows.values():
        for ts in tw:
            if isinstance(ts, pd.Timestamp):
                ref_candidates.append(ts)
    for ts in vessel_available.values():
        if isinstance(ts, pd.Timestamp):
            ref_candidates.append(ts)
    ref_time = min(ref_candidates) if ref_candidates else pd.Timestamp("1970-01-01")

    # Sort voyages by earliest available time (laycan start or voyage start)
    time_rank: Dict[str, float] = {}
    for vk in voyage_keys:
        a_j, _ = time_windows.get(vk, (None, None))
        ts = a_j if isinstance(a_j, pd.Timestamp) else voyage_rows.loc[vk].get(col_start)
        ts = ts if isinstance(ts, pd.Timestamp) else None
        time_rank[vk] = to_days(ts, ref_time) if ts else float("inf")
    sorted_voyages = sorted(voyage_keys, key=lambda k: time_rank.get(k, float("inf")))

    arcs: List[ArcData] = []
    for vessel_key in vessel_keys:
        start_node = f"O_{vessel_key}"
        end_node = f"D_{vessel_key}"
        avail_ts = vessel_available.get(vessel_key)
        avail_days = to_days(avail_ts, ref_time)

        for vk in voyage_keys:
            row = voyage_rows.loc[vk]
            miles = pd.to_numeric(row.get("MILES_DIRECT"), errors="coerce")
            fallback_sail = float(miles) / default_speed_knots / 24.0 if pd.notna(miles) else 1.0
            sail_time = _predict_row(models.sailing_time, row, fallback_value=fallback_sail)
            tau = turnaround_map.get(vk, 1.0)
            cost = sail_time + tau
            _, b_j = time_windows.get(vk, (None, None))
            latest = to_days(b_j, ref_time) + window_slack_days if b_j else None
            arrival = avail_days + sail_time + tau
            if latest is not None and arrival > latest:
                continue
            arcs.append(
                ArcData(
                    vessel=vessel_key,
                    i=start_node,
                    j=vk,
                    sailing_time=sail_time,
                    turnaround_time=tau,
                    cost=cost,
                )
            )
            arcs.append(
                ArcData(
                    vessel=vessel_key,
                    i=vk,
                    j=end_node,
                    sailing_time=0.0,
                    turnaround_time=0.0,
                    cost=0.0,
                )
            )

        # simple nearest-in-time arcs between jobs (capped)
        for idx, i in enumerate(sorted_voyages):
            neighbors = sorted_voyages[idx + 1 : idx + 1 + max_arcs_per_job]
            for j in neighbors:
                row_j = voyage_rows.loc[j]
                miles = pd.to_numeric(row_j.get("MILES_DIRECT"), errors="coerce")
                fallback_sail = float(miles) / default_speed_knots / 24.0 if pd.notna(miles) else 1.0
                sail_time = _predict_row(models.sailing_time, row_j, fallback_value=fallback_sail)
                tau_j = turnaround_map.get(j, 1.0)
                cost = sail_time + tau_j
                a_i, b_i = time_windows.get(i, (None, None))
                depart = a_i or b_i or request.date_window.start_date or avail_ts
                depart_days = to_days(depart, ref_time)
                latest_j = None
                _, b_j = time_windows.get(j, (None, None))
                if b_j is not None:
                    latest_j = to_days(b_j, ref_time) + window_slack_days
                arrival = depart_days + turnaround_map.get(i, 0.0) + sail_time + tau_j
                if latest_j is not None and arrival > latest_j:
                    continue
                arcs.append(
                    ArcData(
                        vessel=vessel_key,
                        i=i,
                        j=j,
                        sailing_time=sail_time,
                        turnaround_time=tau_j,
                        cost=cost,
                    )
                )

    return arcs, turnaround_map, time_windows, vessel_available, unserved_penalty


def solve_mip(
    request: OptimizerRequest,
    models: ModelAdapters,
    max_arcs_per_job: int = 20,
    time_limit_sec: int = 30,
    window_slack_days: float = 2.0,
    default_speed_knots: float = 12.0,
    default_turnaround_days: float = 1.0,
    unserved_penalty: float = 1_000.0,
) -> OptimizerResult:
    arcs, turnaround_map, time_windows, vessel_available, unserved_penalty = build_arcs(
        request,
        models,
        max_arcs_per_job=max_arcs_per_job,
        window_slack_days=window_slack_days,
        default_speed_knots=default_speed_knots,
        default_turnaround_days=default_turnaround_days,
        unserved_penalty=unserved_penalty,
    )
    logs: List[str] = []
    actions: List[OptimizerAction] = []

    if not arcs:
        logs.append("No arcs available; returning empty plan.")
        return OptimizerResult(actions=actions, logs=logs)

    jobs = set(turnaround_map.keys())
    vessels = {arc.vessel for arc in arcs}

    prob = pulp.LpProblem("fleet_plan_mip", pulp.LpMinimize)
    x_vars: Dict[Tuple[str, str, str], pulp.LpVariable] = {}
    t_vars: Dict[str, pulp.LpVariable] = {}
    y_vars: Dict[str, pulp.LpVariable] = {}
    early_vars: Dict[str, pulp.LpVariable] = {}
    late_vars: Dict[str, pulp.LpVariable] = {}

    for arc in arcs:
        key = (arc.vessel, arc.i, arc.j)
        x_vars[key] = pulp.LpVariable(f"x_{arc.vessel}_{arc.i}_{arc.j}", lowBound=0, upBound=1, cat="Binary")

    for job in jobs:
        t_vars[job] = pulp.LpVariable(f"t_{job}", lowBound=0, cat="Continuous")
        y_vars[job] = pulp.LpVariable(f"y_{job}", lowBound=0, upBound=1, cat="Binary")
        early_vars[job] = pulp.LpVariable(f"early_{job}", lowBound=0, cat="Continuous")
        late_vars[job] = pulp.LpVariable(f"late_{job}", lowBound=0, cat="Continuous")

    late_penalty = unserved_penalty * 0.1
    early_penalty = unserved_penalty * 0.01

    prob += (
        pulp.lpSum([arc.cost * x_vars[(arc.vessel, arc.i, arc.j)] for arc in arcs])
        + unserved_penalty * pulp.lpSum([y_vars[j] for j in jobs])
        + late_penalty * pulp.lpSum([late_vars[j] for j in jobs])
        + early_penalty * pulp.lpSum([early_vars[j] for j in jobs])
    )

    arc_index_by_dest = defaultdict(list)
    arc_index_by_src = defaultdict(list)
    for arc in arcs:
        arc_index_by_dest[(arc.vessel, arc.j)].append(arc)
        arc_index_by_src[(arc.vessel, arc.i)].append(arc)

    for job in jobs:
        incoming = [x_vars[(arc.vessel, arc.i, arc.j)] for arc in arcs if arc.j == job]
        prob += pulp.lpSum(incoming) + y_vars[job] == 1, f"assign_or_unserved_{job}"

    for vessel in vessels:
        start_node = f"O_{vessel}"
        end_node = f"D_{vessel}"
        prob += pulp.lpSum([x_vars[(vessel, start_node, arc.j)] for arc in arc_index_by_src[(vessel, start_node)]]) <= 1, f"start_{vessel}"
        prob += pulp.lpSum([x_vars[(vessel, arc.i, end_node)] for arc in arc_index_by_dest[(vessel, end_node)]]) <= 1, f"end_{vessel}"

        for job in jobs:
            incoming = [x_vars[(a.vessel, a.i, a.j)] for a in arc_index_by_dest[(vessel, job)] if a.j == job and a.i != f"O_{vessel}"]
            outgoing = [x_vars[(a.vessel, a.i, a.j)] for a in arc_index_by_src[(vessel, job)] if a.i == job and a.j != f"D_{vessel}"]
            if incoming or outgoing:
                prob += pulp.lpSum(incoming) == pulp.lpSum(outgoing), f"flow_{vessel}_{job}"

    ref_candidates: List[pd.Timestamp] = []
    if isinstance(request.date_window.start_date, pd.Timestamp):
        ref_candidates.append(request.date_window.start_date)
    for tw in time_windows.values():
        for ts in tw:
            if isinstance(ts, pd.Timestamp):
                ref_candidates.append(ts)
    for ts in vessel_available.values():
        if isinstance(ts, pd.Timestamp):
            ref_candidates.append(ts)
    t0 = min(ref_candidates) if ref_candidates else pd.Timestamp("1970-01-01")

    def to_days(ts: Optional[pd.Timestamp]) -> float:
        if not isinstance(ts, pd.Timestamp):
            return 0.0
        return float((ts - t0).total_seconds() / 86400.0)

    if request.date_window.start_date and request.date_window.end_date:
        span_days = to_days(request.date_window.end_date) - to_days(request.date_window.start_date)
        M = max(1.0, span_days + 30.0)
    else:
        M = 365.0

    for arc in arcs:
        if arc.j.startswith("D_"):
            continue
        if arc.i.startswith("O_"):
            t_i = to_days(vessel_available.get(arc.vessel))
        else:
            t_i = t_vars[arc.i]
        t_j = t_vars.get(arc.j)
        if t_j is None:
            continue
        prob += t_j >= t_i + arc.sailing_time + arc.turnaround_time - M * (1 - x_vars[(arc.vessel, arc.i, arc.j)]), f"time_{arc.vessel}_{arc.i}_{arc.j}"

    for job, (a_j, b_j) in time_windows.items():
        if a_j is None and b_j is None:
            continue
        a_val = to_days(a_j)
        b_val = to_days(b_j) if b_j is not None else a_val + M * 0.1
        prob += t_vars[job] >= a_val - early_vars[job], f"tw_lower_{job}"
        prob += t_vars[job] <= b_val + late_vars[job], f"tw_upper_{job}"

    prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit_sec, msg=False))
    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)
    logs.append(f"Solver status: {status}, objective={obj_val:.3f}")

    if status not in ("Optimal", "Not Solved", "Infeasible", "Unbounded", "Undefined") and not actions:
        logs.append("Solver did not return a valid status; falling back to greedy.")
        return GreedyOptimizer(models).plan(request)

    selected_arcs = [arc for arc in arcs if pulp.value(x_vars[(arc.vessel, arc.i, arc.j)]) > 0.5]
    if not selected_arcs:
        logs.append("No arcs selected; falling back to greedy.")
        return GreedyOptimizer(models).plan(request)

    # Build sequences per vessel
    successors: Dict[str, Dict[str, str]] = defaultdict(dict)
    starts: Dict[str, str] = {}
    for arc in selected_arcs:
        if arc.i.startswith("O_"):
            starts[arc.vessel] = arc.j
        successors[arc.vessel][arc.i] = arc.j

    for vessel in successors:
        seq = 1
        current = starts.get(vessel)
        while current and not current.startswith("D_"):
            actions.append(
                AllocateVoyage(
                    fleet_plan_key=request.fleet_plan_key,
                    vessel_key=vessel,
                    voyage_key=current,
                    sequence=seq,
                )
            )
            current = successors[vessel].get(current)
            seq += 1

    logs.append(f"Generated {len(actions)} allocate actions from MIP solution.")
    return OptimizerResult(actions=actions, logs=logs)
