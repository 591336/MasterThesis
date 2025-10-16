"""QA utilities for the port turnaround training dataset.

Generates textual summaries and lightweight SVG charts so artefacts can be
embedded directly in the thesis without heavyweight plotting stacks.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.customer_paths import (
    CustomerPaths,
    describe_customers,
    ensure_customer_dirs,
    resolve_customer,
)
_DEFAULT_PATHS = resolve_customer(None)
ensure_customer_dirs(_DEFAULT_PATHS)
DER = _DEFAULT_PATHS.derived_dir
QA_DIR = DER / "QA"
FIG_DIR = QA_DIR / "figures"
TABLES_DIR = QA_DIR / "tables"
REPORTS_DIR = QA_DIR / "reports"
CURRENT_PATHS = _DEFAULT_PATHS


def configure_customer(slug: str | None) -> CustomerPaths:
    """Update global directories for the selected customer."""
    global DER, QA_DIR, FIG_DIR, TABLES_DIR, REPORTS_DIR, CURRENT_PATHS
    paths = resolve_customer(slug)
    ensure_customer_dirs(paths)
    DER = paths.derived_dir
    QA_DIR = paths.qa_dir
    FIG_DIR = QA_DIR / "figures"
    TABLES_DIR = QA_DIR / "tables"
    REPORTS_DIR = QA_DIR / "reports"
    CURRENT_PATHS = paths
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QA artefacts for port turnaround training data.")
    parser.add_argument(
        "--customer",
        "-c",
        help="Customer slug (e.g. 'northernlights', 'stena'). Defaults to northernlights.",
    )
    parser.add_argument(
        "--list-customers",
        action="store_true",
        help="Print available customer identifiers and exit.",
    )
    return parser.parse_args()


DAYS_IN_PORT_LOWER = 0.04
DAYS_IN_PORT_UPPER = 10.0
KEY_COLUMNS = ["VESSEL_TYPE_ID", "IS_BALLAST", "TERMINAL_ID", "COMMODITY_ID"]
GROUP_COLUMNS = ["PORT_ID", "TERMINAL_ID", "IS_BALLAST"]
MISSINGNESS_HOTSPOT_THRESHOLD = 0.50
MAX_HOTSPOTS_PER_SECTION = 5


@dataclass
class MissingnessTable:
    label: str
    path: Path
    group_cols: Tuple[str, ...]


def load_training() -> pd.DataFrame:
    """Load the derived training CSV."""
    path = DER / "port_turnaround_training.csv"
    return pd.read_csv(path)


def write_text_overview(df: pd.DataFrame) -> None:
    """Persist a human-friendly overview (counts, dtypes, null ratios)."""
    overview_path = REPORTS_DIR / "port_turnaround_training_overview.txt"
    lines = [
        "Port turnaround training dataset - overview",
        f"Rows: {len(df):,}",
        "",
        "Column dtypes:",
    ]
    lines.extend(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
    lines.extend(["", "Null ratios (key columns):"])
    for col in KEY_COLUMNS:
        if col in df:
            ratio = df[col].isna().mean()
            lines.append(f"  {col}: {ratio:.2%}")
        else:
            lines.append(f"  {col}: column missing")
    lines.append("")
    overview_path.write_text("\n".join(lines), encoding="utf-8")


def write_guardrail_stats(df: pd.DataFrame) -> Path:
    """Summarise DAYS_IN_PORT distribution and guardrail breaches."""
    path = REPORTS_DIR / "port_turnaround_training_guardrails.txt"
    if "DAYS_IN_PORT" not in df:
        path.write_text("DAYS_IN_PORT column missing", encoding="utf-8")
        return path

    values = pd.to_numeric(df["DAYS_IN_PORT"], errors="coerce").dropna()
    lines = [
        "DAYS_IN_PORT guardrail check",
        f"Observations (non-null): {len(values):,}",
    ]
    if values.empty:
        lines.append("No non-null values available.")
    else:
        stats = {
            "min": values.min(),
            "p05": values.quantile(0.05),
            "p25": values.quantile(0.25),
            "median": values.quantile(0.5),
            "p75": values.quantile(0.75),
            "p95": values.quantile(0.95),
            "max": values.max(),
        }
        lines.append("")
        lines.append("Key percentiles:")
        for label in ["min", "p05", "p25", "median", "p75", "p95", "max"]:
            lines.append(f"  {label}: {stats[label]:.4f}")
        below = int((values < DAYS_IN_PORT_LOWER).sum())
        above = int((values > DAYS_IN_PORT_UPPER).sum())
        lines.append("")
        lines.append(
            f"Guardrail: {DAYS_IN_PORT_LOWER} <= DAYS_IN_PORT <= {DAYS_IN_PORT_UPPER}"
        )
        lines.append(f"  Below lower bound: {below}")
        lines.append(f"  Above upper bound: {above}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _render_svg_bars(
    title: str,
    categories: Iterable[str],
    values: Iterable[float],
    out_path: Path,
    y_label: str,
    width: int = 840,
    height: int = 360,
) -> None:
    categories = list(categories)
    values = list(values)
    max_value = max(max(values, default=0.0), 1.0)
    margin_x = 80
    margin_y = 70
    plot_width = width - 2 * margin_x
    plot_height = height - 2 * margin_y
    bar_spacing = 10
    bar_width = (plot_width - bar_spacing * max(len(values) - 1, 0)) / max(len(values), 1)

    def bar_rect(idx: int, val: float) -> Tuple[float, float, float, float]:
        scaled_height = 0 if max_value == 0 else (val / max_value) * plot_height
        x = margin_x + idx * (bar_width + bar_spacing)
        y = margin_y + (plot_height - scaled_height)
        return x, y, bar_width, scaled_height

    svg_lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "  <style><![CDATA[text { font-family: 'Helvetica, Arial, sans-serif'; }]]></style>",
        f"  <rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='black' stroke-width='1' />",
        f"  <text x='{width / 2}' y='30' text-anchor='middle' font-size='18'>{title}</text>",
        f"  <text x='{margin_x / 2}' y='{margin_y + plot_height / 2}' text-anchor='middle' font-size='12' transform='rotate(-90 {margin_x / 2} {margin_y + plot_height / 2})'>{y_label}</text>",
        f"  <line x1='{margin_x}' y1='{margin_y + plot_height}' x2='{margin_x + plot_width}' y2='{margin_y + plot_height}' stroke='black' stroke-width='1' />",
        f"  <line x1='{margin_x}' y1='{margin_y}' x2='{margin_x}' y2='{margin_y + plot_height}' stroke='black' stroke-width='1' />",
    ]

    for idx, (cat, val) in enumerate(zip(categories, values)):
        x, y, w, h = bar_rect(idx, val)
        svg_lines.append(
            f"  <rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' fill='#4c78a8' />"
        )
        svg_lines.append(
            f"  <text x='{x + w / 2:.1f}' y='{margin_y + plot_height + 20:.1f}' text-anchor='middle' font-size='12'>{cat}</text>"
        )
        svg_lines.append(
            f"  <text x='{x + w / 2:.1f}' y='{y - 6:.1f}' text-anchor='middle' font-size='11'>{val:.0f}</text>"
        )

    for tick in np.linspace(0, max_value, 5):
        y = margin_y + plot_height - (0 if max_value == 0 else (tick / max_value) * plot_height)
        svg_lines.append(
            f"  <line x1='{margin_x - 5}' y1='{y:.1f}' x2='{margin_x}' y2='{y:.1f}' stroke='black' stroke-width='1' />"
        )
        svg_lines.append(
            f"  <text x='{margin_x - 10}' y='{y + 4:.1f}' text-anchor='end' font-size='11'>{tick:.0f}</text>"
        )

    svg_lines.append("</svg>")
    out_path.write_text("\n".join(svg_lines), encoding="utf-8")


def summarize_missing(
    df: pd.DataFrame,
    available: List[str],
    group_cols: Tuple[str, ...],
) -> pd.DataFrame:
    """Aggregate missingness ratios for the selected columns grouped by group_cols."""
    indicator = df[available].isna().astype(float).add_suffix("_missing_ratio")
    combined = pd.concat([df[list(group_cols)], indicator], axis=1)
    grouped = combined.groupby(list(group_cols), dropna=False)
    summary = grouped.mean(numeric_only=True).reset_index()
    summary["n_obs"] = grouped.size().to_numpy()
    summary.sort_values("n_obs", ascending=False, inplace=True)
    return summary


def write_missingness_tables(
    df: pd.DataFrame,
    available: List[str],
) -> List[MissingnessTable]:
    """Export CSV summaries of missingness grouped by port and by port/terminal."""
    tables: List[MissingnessTable] = []
    if not available or "PORT_ID" not in df.columns:
        path = TABLES_DIR / "port_turnaround_missingness_by_port.csv"
        path.write_text("Required columns missing for missingness summary", encoding="utf-8")
        tables.append(MissingnessTable("by_port", path, ("PORT_ID",)))
        return tables

    by_port = summarize_missing(df, available, ("PORT_ID",))
    by_port_path = TABLES_DIR / "port_turnaround_missingness_by_port.csv"
    by_port.to_csv(by_port_path, index=False)
    tables.append(MissingnessTable("by_port", by_port_path, ("PORT_ID",)))

    if "TERMINAL_ID" in df.columns:
        by_port_terminal = summarize_missing(df, available, ("PORT_ID", "TERMINAL_ID"))
        by_pt_path = TABLES_DIR / "port_turnaround_missingness_by_port_terminal.csv"
        by_port_terminal.to_csv(by_pt_path, index=False)
        tables.append(MissingnessTable("by_port_terminal", by_pt_path, ("PORT_ID", "TERMINAL_ID")))

    return tables


def write_missingness_report(
    tables: List[MissingnessTable],
    available: List[str],
) -> Path:
    """Highlight missingness hotspots above the configured threshold."""
    path = REPORTS_DIR / "port_turnaround_missingness_notes.txt"
    lines = [
        "Missingness hotspots",
        f"Threshold: {MISSINGNESS_HOTSPOT_THRESHOLD:.0%} missing",
        "",
    ]
    if not available:
        lines.append("No key columns available for missingness audit.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def fmt(val: object) -> str:
        if pd.isna(val):
            return "null"
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
        return str(val)

    any_hotspots = False
    for col in available:
        ratio_col = f"{col}_missing_ratio"
        lines.append(f"Column: {col}")
        found_for_col = False
        for table in tables:
            if not table.path.exists():
                continue
            summary = pd.read_csv(table.path)
            if ratio_col not in summary:
                continue
            hotspots = summary[summary[ratio_col] >= MISSINGNESS_HOTSPOT_THRESHOLD]
            if hotspots.empty:
                continue
            any_hotspots = True
            found_for_col = True
            hotspots = hotspots.sort_values(ratio_col, ascending=False).head(MAX_HOTSPOTS_PER_SECTION)
            for _, row in hotspots.iterrows():
                group_desc = ", ".join(
                    f"{gc}={fmt(row.get(gc))}" for gc in table.group_cols
                )
                ratio = row[ratio_col]
                n_obs = int(row.get("n_obs", 0))
                lines.append(
                    f"  - [{table.label}] {group_desc}: {ratio:.0%} missing (n={n_obs})"
                )
        if not found_for_col:
            lines.append("  - No groups exceed threshold")
        lines.append("")

    if not any_hotspots:
        lines.insert(2, "No groups exceeded the missingness threshold.")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def write_histogram(df: pd.DataFrame) -> Path:
    """Generate an SVG histogram for DAYS_IN_PORT."""
    values = df["DAYS_IN_PORT"].dropna().to_numpy()
    bins = np.linspace(0, 10, 11)
    counts, edges = np.histogram(values, bins=bins)
    labels = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(len(edges) - 1)]
    out_path = FIG_DIR / "port_turnaround_days_in_port_hist.svg"
    _render_svg_bars(
        title="DAYS_IN_PORT distribution",
        categories=labels,
        values=counts,
        out_path=out_path,
        y_label="Observations",
    )
    return out_path


def write_missingness_chart(df: pd.DataFrame) -> Path:
    """Generate an SVG bar chart for null ratios in key columns."""
    labels = []
    values = []
    for col in KEY_COLUMNS:
        if col in df:
            labels.append(col)
            values.append(df[col].isna().mean() * 100)
    out_path = FIG_DIR / "port_turnaround_missingness.svg"
    _render_svg_bars(
        title="Missingness by key column",
        categories=labels,
        values=values,
        out_path=out_path,
        y_label="Percent missing",
        height=340,
    )
    return out_path


def write_group_counts(df: pd.DataFrame) -> Path:
    """Summarise coverage per (PORT_ID, TERMINAL_ID, IS_BALLAST)."""
    missing_cols = [col for col in GROUP_COLUMNS if col not in df.columns]
    path = TABLES_DIR / "port_turnaround_group_counts.csv"
    if missing_cols:
        path.write_text(
            f"Required grouping columns missing: {', '.join(missing_cols)}",
            encoding="utf-8",
        )
        return path

    group = df.groupby(GROUP_COLUMNS, dropna=False)
    summary = (
        group["DAYS_IN_PORT"].agg(
            n_obs="size",
            median_days_in_port="median",
            p10_days_in_port=lambda s: s.quantile(0.10),
            p90_days_in_port=lambda s: s.quantile(0.90),
        )
        .reset_index()
        .sort_values("n_obs", ascending=False)
    )
    summary.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    if args.list_customers:
        print(describe_customers())
        return
    paths = configure_customer(args.customer)
    df = load_training()
    if "DAYS_IN_PORT" in df:
        df["DAYS_IN_PORT"] = pd.to_numeric(df["DAYS_IN_PORT"], errors="coerce")

    write_text_overview(df)
    guardrail_path = write_guardrail_stats(df)
    available_key_cols = [col for col in KEY_COLUMNS if col in df.columns]
    missingness_tables = write_missingness_tables(df, available_key_cols)
    missingness_report_path = write_missingness_report(missingness_tables, available_key_cols)
    group_counts_path = write_group_counts(df)
    hist_path = write_histogram(df)
    miss_chart_path = write_missingness_chart(df)

    print(f"Generated artefacts for [{paths.key}]:")
    print(f"  - { (REPORTS_DIR / 'port_turnaround_training_overview.txt').relative_to(ROOT)}")
    print(f"  - {guardrail_path.relative_to(ROOT)}")
    for table in missingness_tables:
        print(f"  - {table.path.relative_to(ROOT)}")
    print(f"  - {missingness_report_path.relative_to(ROOT)}")
    print(f"  - {group_counts_path.relative_to(ROOT)}")
    print(f"  - {hist_path.relative_to(ROOT)}")
    print(f"  - {miss_chart_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
