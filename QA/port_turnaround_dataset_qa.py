"""QA utilities for the port turnaround training dataset.

Generates a quick textual overview and lightweight SVG charts so the artefacts can
be embedded directly in the thesis without relying on heavyweight plotting stacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "DataSets" / "Derived"
QA_DIR = DER / "QA"
FIG_DIR = QA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_training() -> pd.DataFrame:
    """Load the derived training CSV."""
    path = DER / "port_turnaround_training.csv"
    return pd.read_csv(path)


def write_text_overview(df: pd.DataFrame) -> None:
    """Persist a human-friendly overview (counts, dtypes, null ratios)."""
    overview_path = QA_DIR / "port_turnaround_training_overview.txt"
    key_cols = ["VESSEL_TYPE_ID", "IS_BALLAST", "TERMINAL_ID", "COMMODITY_ID"]
    lines = [
        "Port turnaround training dataset — overview",
        f"Rows: {len(df):,}",
        "",
        "Column dtypes:",
    ]
    lines.extend(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
    lines.extend(["", "Null ratios (key columns):"])
    for col in key_cols:
        if col in df:
            ratio = df[col].isna().mean()
            lines.append(f"  {col}: {ratio:.2%}")
        else:
            lines.append(f"  {col}: column missing")
    lines.append("")
    overview_path.write_text("\n".join(lines), encoding="utf-8")


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


def write_histogram(df: pd.DataFrame) -> Path:
    """Generate an SVG histogram for DAYS_IN_PORT."""
    values = df["DAYS_IN_PORT"].dropna().to_numpy()
    bins = np.linspace(0, 10, 11)
    counts, edges = np.histogram(values, bins=bins)
    labels = [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in range(len(edges) - 1)]
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
    key_cols = ["VESSEL_TYPE_ID", "IS_BALLAST", "TERMINAL_ID", "COMMODITY_ID"]
    labels = []
    values = []
    for col in key_cols:
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


def main() -> None:
    df = load_training()
    write_text_overview(df)
    hist_path = write_histogram(df)
    miss_path = write_missingness_chart(df)
    print("Generated artefacts:")
    print(f"  - {hist_path.relative_to(ROOT)}")
    print(f"  - {miss_path.relative_to(ROOT)}")
    print(f"  - {(QA_DIR / 'port_turnaround_training_overview.txt').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
