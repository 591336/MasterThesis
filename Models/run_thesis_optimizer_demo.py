from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Models.evaluate_optimizer_output import check_schedule, manual_overlap  # noqa: E402
from Models.optimizer import ColumnHints, ModelAdapters, OptimizerRequest, ScenarioWindow  # noqa: E402
from Models.optimizer_adapters import SailingTimeModelAdapter, TurnaroundModelAdapter  # noqa: E402
from Models.optimizer_mip import solve_mip  # noqa: E402


@dataclass
class DemoConfig:
    time_limit_sec: int = 60
    max_arcs_per_job: int = 6
    job_job_selection: str = "reposition"
    job_job_candidate_window: int = 30
    unserved_penalty_physical: float = 10_000_000.0
    unserved_penalty_coverage: float = 5_000_000.0

    # Coverage-first laycan settings (used only when hard_laycan_end=False)
    late_penalty_scale: float = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis-ready optimizer artefacts (sample scenario).")
    parser.add_argument("--time-limit-sec", type=int, default=60, help="CBC time limit per run (seconds).")
    parser.add_argument(
        "--output-prefix",
        default="thesis",
        help="Prefix used in Visualizations/output filenames (e.g. allocations_<prefix>_manual_match_physical.csv).",
    )
    parser.add_argument(
        "--require-gates",
        action="store_true",
        help="Fail (non-zero exit) if thesis feasibility gates are not met.",
    )
    parser.add_argument(
        "--overlap-tolerance-sec",
        type=float,
        default=60.0,
        help="Treat overlaps smaller than this threshold as numeric noise (seconds).",
    )
    parser.add_argument(
        "--skip-gantt",
        action="store_true",
        help="Skip generating Gantt PNGs from the exported schedules.",
    )
    return parser.parse_args()


def load_sample() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    base = ROOT / "DataSets" / "Derived" / "Stena" / "SampleScenario"
    vessels = pd.read_csv(base / "vessels_with_availability.csv")
    voyages = pd.read_csv(base / "unallocated_voyages_detailed.csv")
    fleet_plan = pd.read_csv(base / "fleet_plan_voyages.csv")
    manual_path = base / "fleet_plan_manual.csv"
    manual = pd.read_csv(manual_path) if manual_path.exists() else None
    return vessels, voyages, fleet_plan, manual


def prep_dates(vessels: pd.DataFrame, voyages: pd.DataFrame) -> None:
    if "LAYCAN_START_UTC" not in voyages.columns:
        voyages["LAYCAN_START_UTC"] = voyages["ESTIMATED_VOYAGE_START_DATE"]
    if "LAYCAN_END_UTC" not in voyages.columns:
        voyages["LAYCAN_END_UTC"] = voyages["ESTIMATED_VOYAGE_START_DATE"]

    for col in ("ESTIMATED_VOYAGE_START_DATE", "LAYCAN_START_UTC", "LAYCAN_END_UTC"):
        if col in voyages.columns:
            voyages[col] = pd.to_datetime(voyages[col], errors="coerce", utc=True, format="mixed")

    if "VESSEL_AVAILABLE_UTC" in vessels.columns:
        vessels["VESSEL_AVAILABLE_UTC"] = pd.to_datetime(
            vessels["VESSEL_AVAILABLE_UTC"], errors="coerce", utc=True, format="mixed"
        )


def add_haversine_miles(voyages: pd.DataFrame) -> None:
    ports_path = ROOT / "DataSets" / "Derived" / "Stena" / "Static" / "ports_latlon.csv"
    if not ports_path.exists():
        return
    ports = pd.read_csv(ports_path)
    if ports.empty or not {"PORT_ID", "LATITUDE", "LONGITUDE"}.issubset(set(ports.columns)):
        return

    port_map = ports.set_index("PORT_ID")[["LATITUDE", "LONGITUDE"]]

    def haversine_nm(lat1, lon1, lat2, lon2) -> float:
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return 3440.065 * c

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


def load_adapters() -> ModelAdapters:
    sailing_meta = json.loads((ROOT / "DataSets" / "Derived" / "Stena" / "ML" / "sailing_time_features.json").read_text())
    turnaround_meta = json.loads((ROOT / "DataSets" / "Derived" / "Stena" / "ML" / "port_turnaround_features.json").read_text())
    sailing_artifact = ROOT / "Models" / "Artifacts" / "stena" / "sailing_time_dt.joblib"
    turnaround_artifact = ROOT / "Models" / "Artifacts" / "stena" / "port_turnaround_dt.joblib"
    return ModelAdapters(
        turnaround=TurnaroundModelAdapter(turnaround_artifact, turnaround_meta),
        sailing_time=SailingTimeModelAdapter(sailing_artifact, sailing_meta),
    )


def build_request(
    vessels: pd.DataFrame, voyages: pd.DataFrame, fleet_plan: pd.DataFrame, manual: pd.DataFrame | None
) -> OptimizerRequest:
    fleet_plan_key = str(fleet_plan["FLEET_PLAN_ID"].iloc[0]) if not fleet_plan.empty else "unknown"
    hints = ColumnHints(
        vessel_key="VESSEL_ID",
        voyage_key="VOYAGE_ID",
        start_date="ESTIMATED_VOYAGE_START_DATE",
        laycan_start="LAYCAN_START_UTC",
        laycan_end="LAYCAN_END_UTC",
        vessel_available="VESSEL_AVAILABLE_UTC",
    )
    vv = vessels.copy()
    vg = voyages.copy()
    if manual is not None and not manual.empty:
        manual_ids = set(manual["VOYAGE_ID"].astype(str))
        vg = vg[vg["VOYAGE_ID"].astype(str).isin(manual_ids)]
        vessel_ids = set(manual["VESSEL_ID"].astype(str))
        vv = vv[vv["VESSEL_ID"].astype(str).isin(vessel_ids)]

    return OptimizerRequest(
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


def write_outputs(
    prefix: str,
    tag: str,
    result,
    manual: pd.DataFrame | None,
    out_dir: Path,
    overlap_tolerance_sec: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    allocations_path = out_dir / f"allocations_{prefix}_{tag}.csv"
    schedule_path = out_dir / f"schedule_{prefix}_{tag}.csv"
    eval_path = out_dir / f"eval_{prefix}_{tag}.txt"

    allocations_df = pd.DataFrame(
        [
            {
                "fleet_plan_key": a.fleet_plan_key,
                "vessel_key": a.vessel_key,
                "voyage_key": a.voyage_key,
                "sequence": a.sequence,
            }
            for a in result.actions
        ]
    )
    allocations_df.to_csv(allocations_path, index=False)
    if result.schedule is not None and not result.schedule.empty:
        result.schedule.to_csv(schedule_path, index=False)
    else:
        pd.DataFrame().to_csv(schedule_path, index=False)

    sched_stats = check_schedule(pd.read_csv(schedule_path), overlap_tolerance_sec=float(overlap_tolerance_sec))
    manual_stats = manual_overlap(allocations_df, manual)

    lines = [
        f"Run: {tag}",
        f"Status: {getattr(result, 'status', None)}",
        f"Objective: {getattr(result, 'objective', None)}",
        "",
        f"Actions: {sched_stats['n_actions']}",
        f"Vessels used: {sched_stats['n_vessels_used']}",
        f"Max legs per vessel: {sched_stats['max_legs_per_vessel']}",
        f"Overlaps (count): {sched_stats['overlaps']}",
        f"Missing times (start/end): {sched_stats['missing_times']}",
        f"Laycan start violations: {sched_stats['laycan_start_violations']}",
        f"Laycan end violations (start>laycan_end): {sched_stats['laycan_end_violations']}",
        "",
        "Manual overlap:",
        f"  Manual voyages: {manual_stats['manual_total']}",
        f"  Overlap on voyage id: {manual_stats['overlap_any']}",
        f"  Overlap with same vessel: {manual_stats['overlap_vessel_match']}",
        "",
    ]
    eval_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "allocations": str(allocations_path),
        "schedule": str(schedule_path),
        "eval": str(eval_path),
        "sched_stats": sched_stats,
        "manual_stats": manual_stats,
    }


def check_gates(tag: str, stats: dict, strict: bool) -> list[str]:
    failures: list[str] = []
    if stats.get("overlaps", 0) != 0:
        failures.append(f"{tag}: overlaps={stats.get('overlaps')}")
    if stats.get("missing_times", 0) != 0:
        failures.append(f"{tag}: missing_times={stats.get('missing_times')}")
    if strict:
        if stats.get("laycan_start_violations", 0) != 0:
            failures.append(f"{tag}: laycan_start_violations={stats.get('laycan_start_violations')}")
        if stats.get("laycan_end_violations", 0) != 0:
            failures.append(f"{tag}: laycan_end_violations={stats.get('laycan_end_violations')}")
    return failures


def maybe_plot(prefix: str, tag: str, vessels: pd.DataFrame, voyages: pd.DataFrame, schedule_path: Path, out_dir: Path) -> Path | None:
    try:
        from Visualizations.gantt_viz import build_timeline, plot_gantt  # type: ignore
    except Exception:
        return None

    schedule = pd.read_csv(schedule_path, parse_dates=["start", "end", "laycan_start", "laycan_end"])
    timeline = build_timeline(actions=[], vessels=vessels, voyages=voyages, schedule=schedule)
    if timeline.empty:
        return None
    out_path = plot_gantt(timeline, title=f"Optimizer Allocation – {tag}")
    # Copy to a stable name (plot_gantt uses a fixed filename).
    stable = out_dir / f"gantt_{prefix}_{tag}.png"
    stable.write_bytes(Path(out_path).read_bytes())
    return stable


def main() -> None:
    args = parse_args()
    cfg = DemoConfig(time_limit_sec=int(args.time_limit_sec))
    out_dir = ROOT / "Visualizations" / "output"

    vessels, voyages, fleet_plan, manual = load_sample()
    prep_dates(vessels, voyages)
    add_haversine_miles(voyages)
    adapters = load_adapters()
    request = build_request(vessels, voyages, fleet_plan, manual)

    pref = (
        dict(zip(manual["VOYAGE_ID"].astype(str), manual["VESSEL_ID"].astype(str)))
        if manual is not None and not manual.empty
        else None
    )

    # Thesis headline: strict feasibility.
    result_physical = solve_mip(
        request,
        adapters,
        max_arcs_per_job=cfg.max_arcs_per_job,
        time_limit_sec=cfg.time_limit_sec,
        window_slack_days=3.0,
        unserved_penalty=cfg.unserved_penalty_physical,
        hard_laycan_end=True,
        job_job_selection=cfg.job_job_selection,
        job_job_candidate_window=cfg.job_job_candidate_window,
        preferred_vessels=pref,
        switch_penalty=0.5,
    )
    physical_tag = "manual_match_physical"
    physical_outputs = write_outputs(
        args.output_prefix,
        physical_tag,
        result_physical,
        manual,
        out_dir,
        overlap_tolerance_sec=args.overlap_tolerance_sec,
    )
    if not args.skip_gantt:
        maybe_plot(args.output_prefix, physical_tag, request.vessels, request.unallocated_voyages, Path(physical_outputs["schedule"]), out_dir)

    # Contrast run: coverage-first (lateness allowed) to demonstrate trade-off.
    result_coverage = solve_mip(
        request,
        adapters,
        max_arcs_per_job=cfg.max_arcs_per_job,
        time_limit_sec=cfg.time_limit_sec,
        window_slack_days=3.0,
        unserved_penalty=cfg.unserved_penalty_coverage,
        hard_laycan_end=False,
        late_penalty_scale=cfg.late_penalty_scale,
        job_job_selection=cfg.job_job_selection,
        job_job_candidate_window=cfg.job_job_candidate_window,
        preferred_vessels=pref,
        switch_penalty=0.5,
    )
    coverage_tag = "manual_match_coverage"
    coverage_outputs = write_outputs(
        args.output_prefix,
        coverage_tag,
        result_coverage,
        manual,
        out_dir,
        overlap_tolerance_sec=args.overlap_tolerance_sec,
    )
    if not args.skip_gantt:
        maybe_plot(args.output_prefix, coverage_tag, request.vessels, request.unallocated_voyages, Path(coverage_outputs["schedule"]), out_dir)

    gate_failures: list[str] = []
    gate_failures.extend(check_gates(physical_tag, physical_outputs["sched_stats"], strict=True))
    gate_failures.extend(check_gates(coverage_tag, coverage_outputs["sched_stats"], strict=False))

    summary_path = out_dir / f"{args.output_prefix}_demo_summary.txt"
    summary_lines = [
        f"Thesis demo summary ({args.output_prefix})",
        "",
        "Headline (strict):",
        f"  {physical_outputs['eval']}",
        "Contrast (coverage-first):",
        f"  {coverage_outputs['eval']}",
        "",
        "Gates:",
        "  strict: overlaps=0, missing_times=0, laycan_start/end violations=0",
        "  coverage-first: overlaps=0, missing_times=0",
        ("  status: PASS" if not gate_failures else "  status: FAIL"),
        *([f"  - {msg}" for msg in gate_failures] if gate_failures else []),
        "",
        "Commands to reproduce:",
        f"  uv run python Models/run_thesis_optimizer_demo.py --time-limit-sec {cfg.time_limit_sec} --output-prefix {args.output_prefix}",
        "",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    if args.require_gates and gate_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
