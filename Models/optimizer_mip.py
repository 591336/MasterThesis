from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
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


def _to_naive(ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    if not isinstance(ts, pd.Timestamp):
        return None
    if ts.tz is not None:
        return ts.tz_convert(None)
    return ts


def _to_naive_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    try:
        parsed = parsed.dt.tz_convert(None)
    except TypeError:
        # already tz-naive
        pass
    return parsed


@dataclass
class ArcData:
    vessel: str
    i: str
    j: str
    sailing_time: float
    turnaround_time: float
    cost: float
    revenue: float


def _predict_series(model, df: pd.DataFrame, fallback_value: float) -> np.ndarray:
    try:
        # Prefer passing the full frame: adapters may derive features from non-model columns.
        return np.asarray(model.predict(df), dtype=float)
    except Exception:
        try:
            cols = [c for c in getattr(model, "feature_columns", ()) if c in df.columns]
            features = df[cols].copy() if cols else pd.DataFrame(index=df.index)
            return np.asarray(model.predict(features), dtype=float)
        except Exception:
            return np.full(len(df), fallback_value, dtype=float)


def _predict_row(model, row: pd.Series, fallback_value: float) -> float:
    try:
        df = pd.DataFrame([row])
        return float(np.asarray(model.predict(df), dtype=float)[0])
    except Exception:
        try:
            cols = [c for c in getattr(model, "feature_columns", ()) if c in row.index]
            df = pd.DataFrame([row[cols]]) if cols else pd.DataFrame([{}])
            return float(np.asarray(model.predict(df), dtype=float)[0])
        except Exception:
            return fallback_value


def build_arcs(
    request: OptimizerRequest,
    models: ModelAdapters,
    max_arcs_per_job: int = 10,
    window_slack_days: float = 3.0,
    default_speed_knots: float = 12.0,
    default_turnaround_days: float = 1.0,
    cost_scale: float = 1_000_000.0,
    time_weight: float = 0.1,
    preferred_vessels: Optional[Dict[str, str]] = None,
    switch_penalty: float = 0.0,
    min_job_duration_days: float = 0.01,
    job_job_selection: str = "time",
    job_job_candidate_window: Optional[int] = None,
) -> tuple[
    List[ArcData],
    Dict[str, float],
    Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]],
    Dict[str, Optional[pd.Timestamp]],
    Dict[str, int],
]:
    vessels = request.vessels.copy()
    voyages = request.unallocated_voyages.copy()

    col_vessel = request.column_hints.vessel_key
    col_voyage = request.column_hints.voyage_key
    col_start = request.column_hints.start_date
    col_laycan_start = request.column_hints.laycan_start
    col_laycan_end = request.column_hints.laycan_end
    col_vessel_avail = request.column_hints.vessel_available
    col_origin_port = "ORIGIN_PORT_ID"
    col_dest_port = "DEST_PORT_ID"

    voyages[col_voyage] = voyages[col_voyage].astype(str)
    vessels[col_vessel] = vessels[col_vessel].astype(str)

    if col_start and col_start in voyages.columns:
        voyages[col_start] = _to_naive_series(voyages[col_start])
        voyages = voyages.sort_values(col_start, na_position="last")
    if col_laycan_start and col_laycan_start in voyages.columns:
        voyages[col_laycan_start] = _to_naive_series(voyages[col_laycan_start])
    if col_laycan_end and col_laycan_end in voyages.columns:
        voyages[col_laycan_end] = _to_naive_series(voyages[col_laycan_end])
    if col_vessel_avail and col_vessel_avail in vessels.columns:
        vessels[col_vessel_avail] = _to_naive_series(vessels[col_vessel_avail])

    # Load port coordinates if available to compute pairwise distances for sail-time estimates.
    port_map = None
    ports_path = Path("DataSets/Derived/Stena/Static/ports_latlon.csv")
    if ports_path.exists():
        try:
            ports_df = pd.read_csv(ports_path)
            if {"PORT_ID", "LATITUDE", "LONGITUDE"}.issubset(set(ports_df.columns)):
                port_map = ports_df.set_index("PORT_ID")[["LATITUDE", "LONGITUDE"]]
        except Exception:
            port_map = None

    def haversine_nm(lat1, lon1, lat2, lon2) -> float:
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return 3440.065 * c  # nautical miles

    def distance_nm(port_a, port_b) -> Optional[float]:
        if port_map is None:
            return None
        if pd.isna(port_a) or pd.isna(port_b):
            return None
        if port_a not in port_map.index or port_b not in port_map.index:
            return None
        lat1, lon1 = port_map.loc[port_a]
        lat2, lon2 = port_map.loc[port_b]
        return haversine_nm(lat1, lon1, lat2, lon2)

    def sail_time_pair(port_a, port_b) -> Optional[float]:
        dist = distance_nm(port_a, port_b)
        if dist is None:
            return None
        return float(dist / default_speed_knots / 24.0)

    voyage_keys = voyages[col_voyage].astype(str).tolist()
    vessel_keys = vessels[col_vessel].astype(str).tolist()
    if not voyage_keys or not vessel_keys:
        return [], {}, {}, {}, {"start_arcs": 0, "job_job_arcs": 0, "start_arcs_late": 0, "job_job_arcs_late": 0}

    voyage_rows = voyages.set_index(col_voyage)
    vessel_rows = vessels.set_index(col_vessel)

    turnaround_pred = _predict_series(models.turnaround, voyages, fallback_value=default_turnaround_days)
    turnaround_days_map = {
        vk: float(np.clip(tt, 0.0, 10.0)) for vk, tt in zip(voyage_keys, turnaround_pred)
    }

    def fallback_voyage_sailing_days(row: pd.Series) -> float:
        miles = pd.to_numeric(row.get("MILES_TOTAL"), errors="coerce")
        if pd.isna(miles):
            miles = pd.to_numeric(row.get("MILES_DIRECT"), errors="coerce")
        if pd.isna(miles):
            miles_ballast = pd.to_numeric(row.get("MILES_BALLAST"), errors="coerce")
            miles_loaded = pd.to_numeric(row.get("MILES_LOADED"), errors="coerce")
            if pd.notna(miles_ballast) or pd.notna(miles_loaded):
                miles = np.nan_to_num(miles_ballast, nan=0.0) + np.nan_to_num(miles_loaded, nan=0.0)
        if pd.isna(miles) or float(miles) <= 0:
            return 1.0
        return float(miles) / default_speed_knots / 24.0

    sailing_days_map: Dict[str, float] = {}
    job_duration_map: Dict[str, float] = {}
    for vk in voyage_keys:
        row = voyage_rows.loc[vk]
        sail_days = _predict_row(models.sailing_time, row, fallback_value=fallback_voyage_sailing_days(row))
        sail_days = float(max(0.0, sail_days))
        sailing_days_map[vk] = sail_days
        duration = sail_days + turnaround_days_map.get(vk, default_turnaround_days)
        job_duration_map[vk] = float(max(min_job_duration_days, duration))

    time_windows: Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = {}
    for vk in voyage_keys:
        row = voyage_rows.loc[vk]
        a_j = row[col_laycan_start] if col_laycan_start and col_laycan_start in row else None
        b_j = row[col_laycan_end] if col_laycan_end and col_laycan_end in row else None
        a_j = _to_naive(a_j) if isinstance(a_j, pd.Timestamp) else None
        b_j = _to_naive(b_j) if isinstance(b_j, pd.Timestamp) else None
        if a_j is None and b_j is None and col_start and col_start in row:
            a_j = _to_naive(row[col_start]) if isinstance(row[col_start], pd.Timestamp) else None
            b_j = _to_naive(row[col_start]) if isinstance(row[col_start], pd.Timestamp) else None
        time_windows[str(vk)] = (a_j, b_j)

    vessel_available: Dict[str, Optional[pd.Timestamp]] = {}
    for vk in vessel_keys:
        avail = None
        if col_vessel_avail and col_vessel_avail in vessel_rows.columns:
            avail = vessel_rows.loc[vk][col_vessel_avail]
        vessel_available[vk] = _to_naive(avail) if isinstance(avail, pd.Timestamp) else None

    def to_days(ts: Optional[pd.Timestamp], ref: pd.Timestamp) -> float:
        if not isinstance(ts, pd.Timestamp):
            return 0.0
        return float((ts - ref).total_seconds() / 86400.0)

    ref_candidates: List[pd.Timestamp] = []
    if isinstance(request.date_window.start_date, pd.Timestamp):
        start_naive = _to_naive(request.date_window.start_date)
        if start_naive is not None:
            ref_candidates.append(start_naive)
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
    if job_job_candidate_window is None:
        job_job_candidate_window = max_arcs_per_job
    job_job_selection = (job_job_selection or "time").strip().lower()
    if job_job_selection not in {"time", "reposition"}:
        raise ValueError("job_job_selection must be 'time' or 'reposition'")

    arcs: List[ArcData] = []
    arc_stats = {
        "start_arcs": 0,
        "job_job_arcs": 0,
        "start_arcs_late": 0,
        "job_job_arcs_late": 0,
    }
    for vessel_key in vessel_keys:
        start_node = f"O_{vessel_key}"
        end_node = f"D_{vessel_key}"
        avail_ts = vessel_available.get(vessel_key)
        avail_days = to_days(avail_ts, ref_time)
        open_port = None
        if "VESSEL_OPEN_PORT_ID" in vessel_rows.columns:
            open_port = vessel_rows.loc[vessel_key].get("VESSEL_OPEN_PORT_ID")

        for vk in voyage_keys:
            row = voyage_rows.loc[vk]
            origin = row.get(col_origin_port)
            reposition = sail_time_pair(open_port, origin) if open_port is not None else None
            if reposition is None:
                reposition = 0.0
            revenue = pd.to_numeric(row.get("FREIGHT"), errors="coerce")
            gross_freight = pd.to_numeric(row.get("GROSS_FREIGHT"), errors="coerce")
            net_freight = pd.to_numeric(row.get("NET_FREIGHT"), errors="coerce")
            freight_cost = pd.to_numeric(row.get("FREIGHT_COST"), errors="coerce")
            port_cost = pd.to_numeric(row.get("PORT_COST"), errors="coerce")
            handling_cost = pd.to_numeric(row.get("HANDLING_COST"), errors="coerce")
            canal_cost = pd.to_numeric(row.get("VOYAGE_CANAL_COST"), errors="coerce")
            fo_cost = pd.to_numeric(row.get("FO_COST"), errors="coerce")
            do_cost = pd.to_numeric(row.get("DO_COST"), errors="coerce")
            various_cost = pd.to_numeric(row.get("VARIOUS_COST"), errors="coerce")
            various_rev = pd.to_numeric(row.get("VARIOUS_REVENUE"), errors="coerce")
            running_cost = pd.to_numeric(row.get("RUNNING_COST"), errors="coerce")
            revenue_val = np.nan_to_num(net_freight, nan=np.nan_to_num(gross_freight, nan=np.nan_to_num(revenue, nan=0.0)))
            cost_val = 0.0
            for val in (freight_cost, port_cost, handling_cost, canal_cost, fo_cost, do_cost, various_cost, running_cost):
                cost_val += np.nan_to_num(val, nan=0.0)
            cost_component = (cost_val - revenue_val - np.nan_to_num(various_rev, nan=0.0)) / cost_scale  # negative profit term
            cost = cost_component + time_weight * reposition
            if preferred_vessels and preferred_vessels.get(vk) and preferred_vessels.get(vk) != vessel_key:
                cost += switch_penalty
            a_j, b_j = time_windows.get(vk, (None, None))
            arrival_at_origin = avail_days + float(reposition)
            earliest = max(arrival_at_origin, to_days(a_j, ref_time)) if isinstance(a_j, pd.Timestamp) else arrival_at_origin
            latest = to_days(b_j, ref_time) + window_slack_days if b_j else None
            if latest is not None and earliest > latest:
                arc_stats["start_arcs_late"] += 1
            arcs.append(
                ArcData(
                    vessel=vessel_key,
                    i=start_node,
                    j=vk,
                    sailing_time=float(reposition),
                    turnaround_time=0.0,
                    cost=cost,
                    revenue=revenue_val,
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
                    revenue=0.0,
                )
            )
            arc_stats["start_arcs"] += 1

        # simple nearest-in-time arcs between jobs (capped)
        for idx, i in enumerate(sorted_voyages):
            window = sorted_voyages[idx + 1 : idx + 1 + int(job_job_candidate_window)]
            if not window:
                continue
            if job_job_selection == "time":
                neighbors = window[:max_arcs_per_job]
            else:
                row_i = voyage_rows.loc[i]
                dest_i = row_i.get(col_dest_port)

                scored: List[Tuple[float, str]] = []
                for j in window:
                    row_j = voyage_rows.loc[j]
                    origin_j = row_j.get(col_origin_port)
                    pair_sail = sail_time_pair(dest_i, origin_j)
                    if pair_sail is None:
                        pair_sail = 9999.0
                    scored.append((float(pair_sail), j))
                scored.sort(key=lambda t: t[0])
                neighbors = [j for _, j in scored[:max_arcs_per_job]]

            for j in neighbors:
                row_i = voyage_rows.loc[i]
                row_j = voyage_rows.loc[j]
                origin_j = row_j.get(col_origin_port)
                dest_i = row_i.get(col_dest_port)
                pair_sail = sail_time_pair(dest_i, origin_j)
                if pair_sail is None:
                    pair_sail = 0.0
                reposition = float(pair_sail)
                dur_i = job_duration_map.get(i, min_job_duration_days)
                revenue = pd.to_numeric(row_j.get("FREIGHT"), errors="coerce")
                gross_freight = pd.to_numeric(row_j.get("GROSS_FREIGHT"), errors="coerce")
                net_freight = pd.to_numeric(row_j.get("NET_FREIGHT"), errors="coerce")
                freight_cost = pd.to_numeric(row_j.get("FREIGHT_COST"), errors="coerce")
                port_cost = pd.to_numeric(row_j.get("PORT_COST"), errors="coerce")
                handling_cost = pd.to_numeric(row_j.get("HANDLING_COST"), errors="coerce")
                canal_cost = pd.to_numeric(row_j.get("VOYAGE_CANAL_COST"), errors="coerce")
                fo_cost = pd.to_numeric(row_j.get("FO_COST"), errors="coerce")
                do_cost = pd.to_numeric(row_j.get("DO_COST"), errors="coerce")
                various_cost = pd.to_numeric(row_j.get("VARIOUS_COST"), errors="coerce")
                various_rev = pd.to_numeric(row_j.get("VARIOUS_REVENUE"), errors="coerce")
                running_cost = pd.to_numeric(row_j.get("RUNNING_COST"), errors="coerce")
                revenue_val = np.nan_to_num(net_freight, nan=np.nan_to_num(gross_freight, nan=np.nan_to_num(revenue, nan=0.0)))
                cost_val = 0.0
                for val in (freight_cost, port_cost, handling_cost, canal_cost, fo_cost, do_cost, various_cost, running_cost):
                    cost_val += np.nan_to_num(val, nan=0.0)
                cost_component = (cost_val - revenue_val - np.nan_to_num(various_rev, nan=0.0)) / cost_scale
                cost = cost_component + time_weight * reposition
                pref_vessel = preferred_vessels.get(j) if preferred_vessels else None
                if pref_vessel and pref_vessel != vessel_key:
                    cost += switch_penalty
                a_i, b_i = time_windows.get(i, (None, None))
                # Approximate end of job i using its own duration; waiting is handled by the solver via t-vars.
                earliest_i = (
                    to_days(a_i, ref_time)
                    if isinstance(a_i, pd.Timestamp)
                    else to_days(b_i, ref_time)
                    if isinstance(b_i, pd.Timestamp)
                    else avail_days
                )
                end_i = earliest_i + dur_i
                depart_days = (
                    max(end_i, to_days(request.date_window.start_date, ref_time))
                    if isinstance(request.date_window.start_date, pd.Timestamp)
                    else end_i
                )
                latest_j = None
                _, b_j = time_windows.get(j, (None, None))
                if b_j is not None:
                    latest_j = to_days(b_j, ref_time) + window_slack_days
                arrival = depart_days + reposition
                if latest_j is not None and arrival > latest_j:
                    arc_stats["job_job_arcs_late"] += 1
                arcs.append(
                    ArcData(
                        vessel=vessel_key,
                        i=i,
                        j=j,
                        sailing_time=reposition,
                        turnaround_time=dur_i,  # service time at i to prevent overlaps / cycles
                        cost=cost,
                        revenue=revenue_val,
                    )
                )
                arc_stats["job_job_arcs"] += 1

    return arcs, job_duration_map, time_windows, vessel_available, arc_stats


def solve_mip(
    request: OptimizerRequest,
    models: ModelAdapters,
    max_arcs_per_job: int = 20,
    time_limit_sec: int = 30,
    window_slack_days: float = 2.0,
    default_speed_knots: float = 12.0,
    default_turnaround_days: float = 1.0,
    unserved_penalty: float = 1_000.0,
    cost_scale: float = 1_000_000.0,
    enforce_nonoverlap: bool = False,
    time_weight: float = 0.1,
    preferred_vessels: Optional[Dict[str, str]] = None,
    switch_penalty: float = 0.0,
    min_job_duration_days: float = 0.01,
    hard_laycan_end: bool = False,
    late_penalty_scale: float = 0.001,
    max_late_days: Optional[float] = None,
    job_job_selection: str = "time",
    job_job_candidate_window: Optional[int] = None,
    vessel_used_penalty: float = 0.0,
) -> OptimizerResult:
    arcs, job_duration_map, time_windows, vessel_available, arc_stats = build_arcs(
        request,
        models,
        max_arcs_per_job=max_arcs_per_job,
        window_slack_days=window_slack_days,
        default_speed_knots=default_speed_knots,
        default_turnaround_days=default_turnaround_days,
        cost_scale=cost_scale,
        time_weight=time_weight,
        preferred_vessels=preferred_vessels,
        switch_penalty=switch_penalty,
        min_job_duration_days=min_job_duration_days,
        job_job_selection=job_job_selection,
        job_job_candidate_window=job_job_candidate_window,
    )
    logs: List[str] = []
    actions: List[OptimizerAction] = []

    if not arcs:
        logs.append("No arcs available; returning empty plan.")
        return OptimizerResult(actions=actions, logs=logs)
    logs.append(
        f"Arcs total={len(arcs)}, start={arc_stats['start_arcs']} late_start={arc_stats['start_arcs_late']}, "
        f"job_job={arc_stats['job_job_arcs']} late_job_job={arc_stats['job_job_arcs_late']}"
    )

    jobs = set(job_duration_map.keys())
    vessels = {arc.vessel for arc in arcs}

    prob = pulp.LpProblem("fleet_plan_mip", pulp.LpMinimize)
    x_vars: Dict[Tuple[str, str, str], pulp.LpVariable] = {}
    t_vars: Dict[str, pulp.LpVariable] = {}
    y_vars: Dict[str, pulp.LpVariable] = {}
    late_vars: Dict[str, pulp.LpVariable] = {}
    used_vars: Dict[str, pulp.LpVariable] = {}

    for arc in arcs:
        key = (arc.vessel, arc.i, arc.j)
        x_vars[key] = pulp.LpVariable(f"x_{arc.vessel}_{arc.i}_{arc.j}", lowBound=0, upBound=1, cat="Binary")

    for job in jobs:
        t_vars[job] = pulp.LpVariable(f"t_{job}", lowBound=0, cat="Continuous")
        y_vars[job] = pulp.LpVariable(f"y_{job}", lowBound=0, upBound=1, cat="Binary")
        late_vars[job] = pulp.LpVariable(f"late_{job}", lowBound=0, cat="Continuous")

    for vessel in vessels:
        # Keep this continuous: the linking constraints force used_v == start_out (0/1),
        # without introducing additional binary variables that can slow CBC.
        used_vars[vessel] = pulp.LpVariable(f"used_{vessel}", lowBound=0, upBound=1, cat="Continuous")

    late_penalty = unserved_penalty * float(late_penalty_scale)

    prob += (
        pulp.lpSum([arc.cost * x_vars[(arc.vessel, arc.i, arc.j)] for arc in arcs])
        + unserved_penalty * pulp.lpSum([y_vars[j] for j in jobs])
        + (0.0 if hard_laycan_end else late_penalty) * pulp.lpSum([late_vars[j] for j in jobs])
        + float(vessel_used_penalty) * pulp.lpSum([used_vars[v] for v in vessels])
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
        start_out = pulp.lpSum([x_vars[(vessel, start_node, arc.j)] for arc in arc_index_by_src[(vessel, start_node)]])
        prob += start_out <= 1, f"start_{vessel}"
        prob += pulp.lpSum([x_vars[(vessel, arc.i, end_node)] for arc in arc_index_by_dest[(vessel, end_node)]]) <= 1, f"end_{vessel}"
        # Link "used" variable to whether the vessel is activated (any arc leaves start node).
        # With start_out <= 1, these constraints force used_v == start_out.
        prob += used_vars[vessel] >= start_out, f"used_lb_{vessel}"
        prob += used_vars[vessel] <= start_out, f"used_ub_{vessel}"

        for job in jobs:
            incoming = [x_vars[(a.vessel, a.i, a.j)] for a in arc_index_by_dest[(vessel, job)] if a.j == job]
            outgoing = [x_vars[(a.vessel, a.i, a.j)] for a in arc_index_by_src[(vessel, job)] if a.i == job]
            if incoming or outgoing:
                # Degree constraints: a vessel can visit each job at most once, with a single predecessor/successor.
                prob += pulp.lpSum(incoming) <= 1, f"deg_in_{vessel}_{job}"
                prob += pulp.lpSum(outgoing) <= 1, f"deg_out_{vessel}_{job}"
                prob += pulp.lpSum(incoming) == pulp.lpSum(outgoing), f"flow_{vessel}_{job}"

    ref_candidates: List[pd.Timestamp] = []
    if isinstance(request.date_window.start_date, pd.Timestamp):
        start_naive = _to_naive(request.date_window.start_date)
        if start_naive is not None:
            ref_candidates.append(start_naive)
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
    elif ref_candidates:
        days_values = [to_days(ts) for ts in ref_candidates if isinstance(ts, pd.Timestamp)]
        span_days = max(days_values) - min(days_values) if days_values else 0.0
        M = max(365.0, span_days + 60.0)
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
        if a_j is not None:
            prob += t_vars[job] >= to_days(a_j), f"tw_lower_{job}"
        if b_j is not None:
            if hard_laycan_end:
                prob += t_vars[job] <= to_days(b_j), f"tw_upper_{job}"
            else:
                prob += t_vars[job] <= to_days(b_j) + late_vars[job], f"tw_upper_{job}"
                if max_late_days is not None:
                    prob += late_vars[job] <= float(max_late_days), f"tw_latecap_{job}"

    if enforce_nonoverlap:
        # Optional extra disjunctive non-overlap constraints per vessel (quadratic in jobs).
        # Time propagation already prevents cycles when durations are strictly positive.
        assign = {}
        for vessel in vessels:
            for job in jobs:
                incoming = [
                    x_vars[(arc.vessel, arc.i, arc.j)]
                    for arc in arc_index_by_dest[(vessel, job)]
                    if arc.j == job
                ]
                assign[(vessel, job)] = pulp.lpSum(incoming)
        for vessel in vessels:
            job_list = sorted(list(jobs))
            for idx, j1 in enumerate(job_list):
                for j2 in job_list[idx + 1 :]:
                    z = pulp.LpVariable(f"order_{vessel}_{j1}_{j2}", lowBound=0, upBound=1, cat="Binary")
                    dur1 = job_duration_map.get(j1, min_job_duration_days)
                    dur2 = job_duration_map.get(j2, min_job_duration_days)
                    prob += (
                        t_vars[j2]
                        >= t_vars[j1]
                        + dur1
                        - M * (3 - assign[(vessel, j1)] - assign[(vessel, j2)] - z)
                    ), f"nonoverlap1_{vessel}_{j1}_{j2}"
                    prob += (
                        t_vars[j1]
                        >= t_vars[j2]
                        + dur2
                        - M * (2 - assign[(vessel, j1)] - assign[(vessel, j2)] + z)
                    ), f"nonoverlap2_{vessel}_{j1}_{j2}"

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

    def build_schedule() -> Optional[pd.DataFrame]:
        if not actions:
            return None

        schedule_rows: List[dict] = []

        # If CBC reports Optimal, use t-vars as-is (they should be consistent with constraints).
        if status == "Optimal":
            for action in actions:
                job = str(action.voyage_key)
                t_days = pulp.value(t_vars.get(job))
                start_ts = t0 + pd.to_timedelta(float(t_days), unit="D") if t_days is not None else pd.NaT
                if isinstance(start_ts, pd.Timestamp):
                    start_ts = start_ts.round("s")
                duration_days = float(job_duration_map.get(job, min_job_duration_days))
                end_ts = (
                    start_ts + pd.to_timedelta(duration_days, unit="D")
                    if isinstance(start_ts, pd.Timestamp)
                    else pd.NaT
                )
                if isinstance(end_ts, pd.Timestamp):
                    end_ts = end_ts.round("s")
                a_j, b_j = time_windows.get(job, (None, None))
                schedule_rows.append(
                    {
                        "fleet_plan_key": request.fleet_plan_key,
                        "vessel_key": str(action.vessel_key),
                        "voyage_key": job,
                        "sequence": int(action.sequence),
                        "start": start_ts,
                        "end": end_ts,
                        "duration_days": duration_days,
                        "laycan_start": a_j,
                        "laycan_end": b_j,
                    }
                )
        else:
            # If the solver did not prove optimality (e.g. time limit), build a consistent
            # schedule by forward-propagating each vessel sequence using the time windows.
            vessels_df = request.vessels.copy()
            col_vessel = request.column_hints.vessel_key
            col_vessel_avail = request.column_hints.vessel_available
            vessels_df[col_vessel] = vessels_df[col_vessel].astype(str)
            vessels_df = vessels_df.set_index(col_vessel, drop=False)
            if col_vessel_avail and col_vessel_avail in vessels_df.columns:
                vessels_df[col_vessel_avail] = _to_naive_series(vessels_df[col_vessel_avail])

            def vessel_available_ts(vessel_key: str) -> Optional[pd.Timestamp]:
                if col_vessel_avail and col_vessel_avail in vessels_df.columns and vessel_key in vessels_df.index:
                    ts = vessels_df.loc[vessel_key].get(col_vessel_avail)
                    return ts if isinstance(ts, pd.Timestamp) else None
                return None

            for vessel_key, vessel_actions in pd.DataFrame(
                [{"vessel": a.vessel_key, "voyage": a.voyage_key, "seq": a.sequence} for a in actions]
            ).groupby("vessel"):
                ordered = vessel_actions.sort_values("seq")
                current_ts = vessel_available_ts(str(vessel_key)) or t0
                for _, row in ordered.iterrows():
                    job = str(row["voyage"])
                    a_j, b_j = time_windows.get(job, (None, None))
                    if isinstance(a_j, pd.Timestamp):
                        start_ts = max(current_ts, a_j)
                    else:
                        start_ts = current_ts
                    duration_days = float(job_duration_map.get(job, min_job_duration_days))
                    end_ts = start_ts + pd.to_timedelta(duration_days, unit="D")
                    schedule_rows.append(
                        {
                            "fleet_plan_key": request.fleet_plan_key,
                            "vessel_key": str(vessel_key),
                            "voyage_key": job,
                            "sequence": int(row["seq"]),
                            "start": start_ts.round("s"),
                            "end": end_ts.round("s"),
                            "duration_days": duration_days,
                            "laycan_start": a_j,
                            "laycan_end": b_j,
                        }
                    )
                    current_ts = end_ts

        if not schedule_rows:
            return None
        return pd.DataFrame(schedule_rows).sort_values(["vessel_key", "sequence"]).reset_index(drop=True)

    schedule = build_schedule()
    return OptimizerResult(actions=actions, logs=logs, schedule=schedule, status=status, objective=float(obj_val) if obj_val is not None else None)
