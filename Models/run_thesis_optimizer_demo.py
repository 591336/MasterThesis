from __future__ import annotations

import argparse
import json
import re
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


_MISSING_REPOSITION_PATTERN = re.compile(
    r"missing_reposition_start=(?P<start>\d+)\s+missing_reposition_job_job=(?P<job_job>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis-ready optimizer artefacts (sample scenario).")
    parser.add_argument("--time-limit-sec", type=int, default=60, help="CBC time limit per run (seconds).")
    parser.add_argument(
        "--missing-reposition-days",
        type=float,
        default=1.0,
        help="Conservative fallback reposition time (days) when port pair distance is unavailable.",
    )
    parser.add_argument(
        "--vessel-used-penalty",
        type=float,
        default=None,
        help="Optional fixed penalty per activated vessel (legacy: sets both strict+coverage penalties).",
    )
    parser.add_argument(
        "--vessel-used-penalty-strict",
        type=float,
        default=0.0,
        help="Fixed penalty per activated vessel for the strict run (discourages excessive vessel usage).",
    )
    parser.add_argument(
        "--vessel-used-penalty-coverage",
        type=float,
        default=0.0,
        help="Fixed penalty per activated vessel for the coverage run (use with care; can change the trade-off).",
    )
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
    parser.add_argument(
        "--min-dest-port-coverage",
        type=float,
        default=0.90,
        help="Minimum acceptable DEST_PORT_ID non-null coverage (0-1) for the effective voyage set.",
    )
    parser.add_argument(
        "--require-dest-port-coverage",
        action="store_true",
        help="Fail fast when effective DEST_PORT_ID coverage is below --min-dest-port-coverage.",
    )
    parser.add_argument(
        "--turnaround-model",
        choices=("dt", "hgbt"),
        default="dt",
        help="Turnaround model artifact to use for the optimisation runs.",
    )
    parser.add_argument(
        "--sailing-model",
        choices=("dt",),
        default="dt",
        help="Sailing-time model artifact to use for the optimisation runs.",
    )
    return parser.parse_args()


def load_sample() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    base = ROOT / "DataSets" / "Derived" / "Customer1" / "SampleScenario"
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
    ports_path = ROOT / "DataSets" / "Derived" / "Customer1" / "Static" / "ports_latlon.csv"
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


def summarize_port_coverage(voyages: pd.DataFrame) -> dict:
    n_rows = int(len(voyages))
    if n_rows == 0:
        return {
            "rows": 0,
            "origin_nonnull": 0,
            "origin_nonnull_pct": 0.0,
            "dest_nonnull": 0,
            "dest_nonnull_pct": 0.0,
            "origin_mapped": None,
            "origin_mapped_pct_rows": None,
            "dest_mapped": None,
            "dest_mapped_pct_rows": None,
        }

    origin = pd.to_numeric(voyages.get("ORIGIN_PORT_ID"), errors="coerce")
    dest = pd.to_numeric(voyages.get("DEST_PORT_ID"), errors="coerce")
    origin_nonnull = int(origin.notna().sum())
    dest_nonnull = int(dest.notna().sum())

    summary = {
        "rows": n_rows,
        "origin_nonnull": origin_nonnull,
        "origin_nonnull_pct": float(origin_nonnull / n_rows),
        "dest_nonnull": dest_nonnull,
        "dest_nonnull_pct": float(dest_nonnull / n_rows),
        "origin_mapped": None,
        "origin_mapped_pct_rows": None,
        "dest_mapped": None,
        "dest_mapped_pct_rows": None,
    }

    ports_path = ROOT / "DataSets" / "Derived" / "Customer1" / "Static" / "ports_latlon.csv"
    if not ports_path.exists():
        return summary
    ports = pd.read_csv(ports_path)
    if ports.empty or "PORT_ID" not in ports.columns:
        return summary

    port_ids = set(pd.to_numeric(ports["PORT_ID"], errors="coerce").dropna().astype(int).tolist())
    origin_mapped = int(origin.dropna().astype(int).isin(port_ids).sum())
    dest_mapped = int(dest.dropna().astype(int).isin(port_ids).sum())
    summary["origin_mapped"] = origin_mapped
    summary["origin_mapped_pct_rows"] = float(origin_mapped / n_rows)
    summary["dest_mapped"] = dest_mapped
    summary["dest_mapped_pct_rows"] = float(dest_mapped / n_rows)
    return summary


def load_adapters(turnaround_model: str = "dt", sailing_model: str = "dt") -> ModelAdapters:
    sailing_meta = json.loads((ROOT / "DataSets" / "Derived" / "Customer1" / "ML" / "sailing_time_features.json").read_text())
    turnaround_meta = json.loads((ROOT / "DataSets" / "Derived" / "Customer1" / "ML" / "port_turnaround_features.json").read_text())
    sailing_artifact = ROOT / "Models" / "Artifacts" / "customer1" / f"sailing_time_{sailing_model}.joblib"
    turnaround_artifact = ROOT / "Models" / "Artifacts" / "customer1" / f"port_turnaround_{turnaround_model}.joblib"
    if not sailing_artifact.exists():
        raise FileNotFoundError(f"Missing sailing artifact: {sailing_artifact}")
    if not turnaround_artifact.exists():
        raise FileNotFoundError(f"Missing turnaround artifact: {turnaround_artifact}")
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
        scenario_key="customer1_sample",
        scenario_code="customer1_sample",
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
        schedule_df = result.schedule.copy()
    else:
        schedule_df = pd.DataFrame(
            columns=["fleet_plan_key", "vessel_key", "voyage_key", "start", "end", "laycan_start", "laycan_end"]
        )
    schedule_df.to_csv(schedule_path, index=False)

    sched_stats = check_schedule(schedule_df, overlap_tolerance_sec=float(overlap_tolerance_sec))
    manual_stats = manual_overlap(allocations_df, manual)
    reposition_diag = extract_reposition_diagnostics(result)

    start_missing_txt = (
        str(reposition_diag["start_missing_reposition"])
        if reposition_diag["start_missing_reposition"] is not None
        else "not available"
    )
    job_job_missing_txt = (
        str(reposition_diag["job_job_missing_reposition"])
        if reposition_diag["job_job_missing_reposition"] is not None
        else "not available"
    )

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
        f"Missing reposition arcs (start): {start_missing_txt}",
        f"Missing reposition arcs (job_job): {job_job_missing_txt}",
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
        "reposition_diag": reposition_diag,
    }


def extract_reposition_diagnostics(result) -> dict[str, int | None]:
    logs = list(getattr(result, "logs", []) or [])
    for line in logs:
        match = _MISSING_REPOSITION_PATTERN.search(str(line))
        if match:
            return {
                "start_missing_reposition": int(match.group("start")),
                "job_job_missing_reposition": int(match.group("job_job")),
            }
    return {"start_missing_reposition": None, "job_job_missing_reposition": None}


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

    strict_vessel_penalty = float(args.vessel_used_penalty_strict)
    coverage_vessel_penalty = float(args.vessel_used_penalty_coverage)
    if args.vessel_used_penalty is not None:
        strict_vessel_penalty = float(args.vessel_used_penalty)
        coverage_vessel_penalty = float(args.vessel_used_penalty)

    vessels, voyages, fleet_plan, manual = load_sample()
    prep_dates(vessels, voyages)
    add_haversine_miles(voyages)
    adapters = load_adapters(turnaround_model=args.turnaround_model, sailing_model=args.sailing_model)
    request = build_request(vessels, voyages, fleet_plan, manual)
    if manual is not None and not manual.empty and request.unallocated_voyages.empty:
        manual_n = int(manual["VOYAGE_ID"].astype("string").nunique())
        candidate_n = int(voyages["VOYAGE_ID"].astype("string").nunique())
        raise SystemExit(
            "Manual benchmark mismatch: zero voyage-id overlap between "
            f"fleet_plan_manual.csv (n={manual_n}) and unallocated_voyages_detailed.csv (n={candidate_n}). "
            "Export voyages for the matching scenario/fleet-plan before running thesis demo."
        )
    full_cov = summarize_port_coverage(voyages)
    eff_cov = summarize_port_coverage(request.unallocated_voyages)
    min_dest_cov = float(max(0.0, min(1.0, args.min_dest_port_coverage)))
    if args.require_dest_port_coverage and eff_cov["dest_nonnull_pct"] < min_dest_cov:
        raise SystemExit(
            "DEST_PORT_ID coverage gate failed: "
            f"effective coverage={eff_cov['dest_nonnull_pct']:.1%} < required={min_dest_cov:.1%}. "
            "Refresh unallocated_voyages_detailed.csv before running."
        )

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
        vessel_used_penalty=strict_vessel_penalty,
        missing_reposition_days=float(args.missing_reposition_days),
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
        vessel_used_penalty=coverage_vessel_penalty,
        missing_reposition_days=float(args.missing_reposition_days),
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
    strict_diag = physical_outputs.get("reposition_diag", {})
    coverage_diag = coverage_outputs.get("reposition_diag", {})

    strict_start_txt = (
        str(strict_diag.get("start_missing_reposition"))
        if strict_diag.get("start_missing_reposition") is not None
        else "not available"
    )
    strict_job_job_txt = (
        str(strict_diag.get("job_job_missing_reposition"))
        if strict_diag.get("job_job_missing_reposition") is not None
        else "not available"
    )
    coverage_start_txt = (
        str(coverage_diag.get("start_missing_reposition"))
        if coverage_diag.get("start_missing_reposition") is not None
        else "not available"
    )
    coverage_job_job_txt = (
        str(coverage_diag.get("job_job_missing_reposition"))
        if coverage_diag.get("job_job_missing_reposition") is not None
        else "not available"
    )

    summary_lines = [
        f"Thesis demo summary ({args.output_prefix})",
        "",
        "Port ID coverage (full input):",
        f"  rows: {full_cov['rows']}",
        f"  ORIGIN_PORT_ID non-null: {full_cov['origin_nonnull']}/{full_cov['rows']} ({full_cov['origin_nonnull_pct']:.1%})",
        f"  DEST_PORT_ID non-null: {full_cov['dest_nonnull']}/{full_cov['rows']} ({full_cov['dest_nonnull_pct']:.1%})",
        *(
            [
                f"  ORIGIN_PORT_ID mapped rows: {full_cov['origin_mapped']}/{full_cov['rows']} ({full_cov['origin_mapped_pct_rows']:.1%})",
                f"  DEST_PORT_ID mapped rows: {full_cov['dest_mapped']}/{full_cov['rows']} ({full_cov['dest_mapped_pct_rows']:.1%})",
            ]
            if full_cov["origin_mapped"] is not None
            else []
        ),
        "",
        "Port ID coverage (effective optimiser set):",
        f"  rows: {eff_cov['rows']}",
        f"  ORIGIN_PORT_ID non-null: {eff_cov['origin_nonnull']}/{eff_cov['rows']} ({eff_cov['origin_nonnull_pct']:.1%})",
        f"  DEST_PORT_ID non-null: {eff_cov['dest_nonnull']}/{eff_cov['rows']} ({eff_cov['dest_nonnull_pct']:.1%})",
        *(
            [
                f"  ORIGIN_PORT_ID mapped rows: {eff_cov['origin_mapped']}/{eff_cov['rows']} ({eff_cov['origin_mapped_pct_rows']:.1%})",
                f"  DEST_PORT_ID mapped rows: {eff_cov['dest_mapped']}/{eff_cov['rows']} ({eff_cov['dest_mapped_pct_rows']:.1%})",
            ]
            if eff_cov["origin_mapped"] is not None
            else []
        ),
        (
            f"  DEST coverage gate: FAIL ({eff_cov['dest_nonnull_pct']:.1%} < {min_dest_cov:.1%})"
            if eff_cov["dest_nonnull_pct"] < min_dest_cov
            else f"  DEST coverage gate: PASS ({eff_cov['dest_nonnull_pct']:.1%} >= {min_dest_cov:.1%})"
        ),
        "",
        "Headline (strict):",
        f"  {physical_outputs['eval']}",
        "Contrast (coverage-first):",
        f"  {coverage_outputs['eval']}",
        "Objective framing:",
        "  simplified economic proxy objective (not a full calibrated commercial P&L)",
        "Model artifacts:",
        f"  turnaround: Models/Artifacts/customer1/port_turnaround_{args.turnaround_model}.joblib",
        f"  sailing_time: Models/Artifacts/customer1/sailing_time_{args.sailing_model}.joblib",
        "Missing reposition diagnostics (strict):",
        f"  start_missing_reposition={strict_start_txt}, job_job_missing_reposition={strict_job_job_txt}",
        "Missing reposition diagnostics (coverage-first):",
        f"  start_missing_reposition={coverage_start_txt}, job_job_missing_reposition={coverage_job_job_txt}",
        "",
        f"Missing reposition fallback (days): {float(args.missing_reposition_days)}",
        f"Vessel used penalty (strict): {strict_vessel_penalty}",
        f"Vessel used penalty (coverage): {coverage_vessel_penalty}",
        "Gates:",
        "  strict: overlaps=0, missing_times=0, laycan_start/end violations=0",
        "  coverage-first: overlaps=0, missing_times=0",
        ("  status: PASS" if not gate_failures else "  status: FAIL"),
        *([f"  - {msg}" for msg in gate_failures] if gate_failures else []),
        "",
        "Commands to reproduce:",
        f"  uv run python Models/run_thesis_optimizer_demo.py --time-limit-sec {cfg.time_limit_sec} --output-prefix {args.output_prefix} --missing-reposition-days {float(args.missing_reposition_days)} --vessel-used-penalty-strict {strict_vessel_penalty} --vessel-used-penalty-coverage {coverage_vessel_penalty} --turnaround-model {args.turnaround_model} --sailing-model {args.sailing_model}",
        "",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    if args.require_gates and gate_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
