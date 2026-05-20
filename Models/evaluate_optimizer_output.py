from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class EvalPaths:
    sample_dir: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate optimizer allocations/schedule for the sample scenario.")
    parser.add_argument(
        "--experiment",
        default="subset_ml_proxy",
        help="Experiment name used in Visualizations/output/{allocations|schedule}_<name>.csv",
    )
    parser.add_argument(
        "--sample-dir",
        default="DataSets/Derived/Customer1/SampleScenario",
        help="Directory containing the sample scenario CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default="Visualizations/output",
        help="Directory containing optimizer outputs.",
    )
    parser.add_argument(
        "--overlap-tolerance-sec",
        type=float,
        default=60.0,
        help="Treat overlaps smaller than this threshold as numeric noise (seconds).",
    )
    return parser.parse_args()


def load_inputs(paths: EvalPaths, experiment: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    allocations_path = paths.output_dir / f"allocations_{experiment}.csv"
    schedule_path = paths.output_dir / f"schedule_{experiment}.csv"
    if not allocations_path.exists():
        raise FileNotFoundError(f"Missing allocations file: {allocations_path}")
    if not schedule_path.exists():
        raise FileNotFoundError(f"Missing schedule file: {schedule_path}")

    allocations = pd.read_csv(allocations_path, dtype=str)
    schedule = pd.read_csv(
        schedule_path,
        parse_dates=["start", "end", "laycan_start", "laycan_end"],
        dtype={"fleet_plan_key": "string", "vessel_key": "string", "voyage_key": "string"},
    )

    voyages = pd.read_csv(paths.sample_dir / "unallocated_voyages_detailed.csv", dtype=str)
    manual_path = paths.sample_dir / "fleet_plan_manual.csv"
    manual = pd.read_csv(manual_path, dtype=str) if manual_path.exists() else None
    return allocations, schedule, voyages, manual


def check_schedule(schedule: pd.DataFrame, overlap_tolerance_sec: float) -> dict:
    if schedule.empty:
        return {
            "n_actions": 0,
            "n_vessels_used": 0,
            "max_legs_per_vessel": 0,
            "overlaps": 0,
            "missing_times": 0,
            "laycan_start_violations": 0,
            "laycan_end_violations": 0,
        }

    sched = schedule.copy()
    sched["vessel_key"] = sched["vessel_key"].astype("string")
    sched["voyage_key"] = sched["voyage_key"].astype("string")

    for col in ("start", "end", "laycan_start", "laycan_end"):
        if col in sched.columns:
            sched[col] = (
                pd.to_datetime(sched[col], errors="coerce", utc=True, format="mixed")
                .dt.tz_convert(None)
            )

    missing_times = int(sched["start"].isna().sum() + sched["end"].isna().sum())

    overlaps = 0
    tol = pd.Timedelta(seconds=float(overlap_tolerance_sec))
    for vessel, group in sched.dropna(subset=["start", "end"]).groupby("vessel_key"):
        group = group.sort_values("start")
        prev_end = None
        for _, row in group.iterrows():
            start = row["start"]
            end = row["end"]
            if prev_end is not None and isinstance(start, pd.Timestamp) and start + tol < prev_end:
                overlaps += 1
            prev_end = end

    laycan_start_violations = int(
        ((sched["laycan_start"].notna()) & (sched["start"].notna()) & (sched["start"] < sched["laycan_start"])).sum()
    )
    laycan_end_violations = int(
        ((sched["laycan_end"].notna()) & (sched["start"].notna()) & (sched["start"] > sched["laycan_end"])).sum()
    )

    legs_per_vessel = sched.groupby("vessel_key")["voyage_key"].size() if not sched.empty else pd.Series(dtype=int)
    return {
        "n_actions": int(len(sched)),
        "n_vessels_used": int(sched["vessel_key"].nunique()),
        "max_legs_per_vessel": int(legs_per_vessel.max()) if not legs_per_vessel.empty else 0,
        "overlaps": int(overlaps),
        "missing_times": int(missing_times),
        "laycan_start_violations": int(laycan_start_violations),
        "laycan_end_violations": int(laycan_end_violations),
    }


def manual_overlap(allocations: pd.DataFrame, manual: Optional[pd.DataFrame]) -> dict:
    if manual is None or manual.empty or allocations.empty:
        return {"manual_total": int(len(manual) if manual is not None else 0), "overlap_vessel_match": 0, "overlap_any": 0}

    alloc = allocations.copy()
    alloc["voyage_key"] = alloc["voyage_key"].astype("string")
    alloc["vessel_key"] = alloc["vessel_key"].astype("string")
    man = manual.copy()
    man["VOYAGE_ID"] = man["VOYAGE_ID"].astype("string")
    man["VESSEL_ID"] = man["VESSEL_ID"].astype("string")

    merged = alloc.merge(man, left_on="voyage_key", right_on="VOYAGE_ID", how="inner")
    overlap_any = int(len(merged))
    overlap_vessel_match = int((merged["vessel_key"] == merged["VESSEL_ID"]).sum())
    return {
        "manual_total": int(len(man)),
        "overlap_any": overlap_any,
        "overlap_vessel_match": overlap_vessel_match,
    }


def main() -> None:
    args = parse_args()
    paths = EvalPaths(sample_dir=Path(args.sample_dir), output_dir=Path(args.output_dir))

    allocations, schedule, voyages, manual = load_inputs(paths, args.experiment)
    sched_stats = check_schedule(schedule, overlap_tolerance_sec=args.overlap_tolerance_sec)
    manual_stats = manual_overlap(allocations, manual)

    report_lines = [
        f"Experiment: {args.experiment}",
        f"Actions: {sched_stats['n_actions']}",
        f"Vessels used: {sched_stats['n_vessels_used']}",
        f"Max legs per vessel: {sched_stats['max_legs_per_vessel']}",
        f"Overlaps (count): {sched_stats['overlaps']}",
        f"Missing times (start/end): {sched_stats['missing_times']}",
        f"Laycan start violations: {sched_stats['laycan_start_violations']}",
        f"Laycan end violations (start>laycan_end): {sched_stats['laycan_end_violations']}",
    ]
    if manual is not None:
        report_lines.extend(
            [
                "",
                "Manual overlap:",
                f"  Manual voyages: {manual_stats['manual_total']}",
                f"  Overlap on voyage id: {manual_stats['overlap_any']}",
                f"  Overlap with same vessel: {manual_stats['overlap_vessel_match']}",
            ]
        )

    report = "\n".join(report_lines) + "\n"
    out_path = paths.output_dir / f"eval_{args.experiment}.txt"
    out_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
