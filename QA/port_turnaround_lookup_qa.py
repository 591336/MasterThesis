"""QA visuals for the port turnaround lookup table.

Produces a combined SVG summarising duration bins and fallback level coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "DataSets" / "Derived"
QA_DIR = DER / "QA"
FIG_DIR = QA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_lookup_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    lookup = pd.read_csv(DER / "port_turnaround_lookup.csv")
    qa_bins = pd.read_csv(DER / "port_turnaround_lookup_qa_bins.csv")
    return lookup, qa_bins


def _panel_geometry(width: int, height: int, n_panels: int = 2, gutter: int = 40):
    margin_y = 70
    usable_width = width - gutter * (n_panels + 1)
    panel_width = usable_width / n_panels
    origins = []
    for idx in range(n_panels):
        x0 = gutter * (idx + 1) + panel_width * idx
        origins.append((x0, margin_y))
    plot_height = height - 2 * margin_y
    return panel_width, plot_height, origins


def _render_panel(
    categories: Iterable[str],
    values: Iterable[float],
    title: str,
    y_label: str,
    panel_width: float,
    plot_height: float,
    origin_x: float,
    origin_y: float,
) -> list[str]:
    categories = list(categories)
    values = list(values)
    max_value = max(max(values, default=0.0), 1.0)
    bar_spacing = 10
    bar_width = (panel_width - bar_spacing * max(len(values) - 1, 0)) / max(len(values), 1)

    lines = [
        f"  <text x='{origin_x + panel_width / 2:.1f}' y='{origin_y - 30:.1f}' text-anchor='middle' font-size='18'>{title}</text>",
        f"  <text x='{origin_x - 40:.1f}' y='{origin_y + plot_height / 2:.1f}' text-anchor='middle' font-size='12' transform='rotate(-90 {origin_x - 40:.1f} {origin_y + plot_height / 2:.1f})'>{y_label}</text>",
        f"  <line x1='{origin_x:.1f}' y1='{origin_y + plot_height:.1f}' x2='{origin_x + panel_width:.1f}' y2='{origin_y + plot_height:.1f}' stroke='black' stroke-width='1' />",
        f"  <line x1='{origin_x:.1f}' y1='{origin_y:.1f}' x2='{origin_x:.1f}' y2='{origin_y + plot_height:.1f}' stroke='black' stroke-width='1' />",
    ]

    for idx, (cat, val) in enumerate(zip(categories, values)):
        scaled_height = 0 if max_value == 0 else (val / max_value) * plot_height
        x = origin_x + idx * (bar_width + bar_spacing)
        y = origin_y + (plot_height - scaled_height)
        lines.append(
            f"  <rect x='{x:.1f}' y='{y:.1f}' width='{bar_width:.1f}' height='{scaled_height:.1f}' fill='#4c78a8' />"
        )
        lines.append(
            f"  <text x='{x + bar_width / 2:.1f}' y='{origin_y + plot_height + 20:.1f}' text-anchor='middle' font-size='12'>{cat}</text>"
        )
        lines.append(
            f"  <text x='{x + bar_width / 2:.1f}' y='{y - 6:.1f}' text-anchor='middle' font-size='11'>{val:.0f}</text>"
        )

    for tick in range(5):
        frac = tick / 4 if 4 else 0
        tick_val = frac * max_value
        y = origin_y + plot_height - frac * plot_height
        lines.append(
            f"  <line x1='{origin_x - 5:.1f}' y1='{y:.1f}' x2='{origin_x:.1f}' y2='{y:.1f}' stroke='black' stroke-width='1' />"
        )
        lines.append(
            f"  <text x='{origin_x - 10:.1f}' y='{y + 4:.1f}' text-anchor='end' font-size='11'>{tick_val:.0f}</text>"
        )

    return lines


def write_combined_lookup_svg(lookup: pd.DataFrame, qa_bins: pd.DataFrame) -> Path:
    width = 960
    height = 360
    panel_width, plot_height, origins = _panel_geometry(width, height, n_panels=2)

    level_counts = lookup.groupby("level").size().sort_index()
    level_labels = [f"L{lvl}" for lvl in level_counts.index]
    level_values = level_counts.to_list()

    bin_labels = qa_bins["BIN"].tolist()
    bin_values = qa_bins["n"].tolist()

    svg_lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "  <style><![CDATA[text { font-family: 'Helvetica, Arial, sans-serif'; }]]></style>",
        "  <rect x='0' y='0' width='100%' height='100%' fill='white' stroke='black' stroke-width='1' />",
    ]

    svg_lines.extend(
        _render_panel(
            categories=bin_labels,
            values=bin_values,
            title="DAYS_IN_PORT bins",
            y_label="Observations",
            panel_width=panel_width,
            plot_height=plot_height,
            origin_x=origins[0][0],
            origin_y=origins[0][1],
        )
    )

    svg_lines.extend(
        _render_panel(
            categories=level_labels,
            values=level_values,
            title="Lookup rows per level",
            y_label="Row count",
            panel_width=panel_width,
            plot_height=plot_height,
            origin_x=origins[1][0],
            origin_y=origins[1][1],
        )
    )

    svg_lines.append("</svg>")
    out_path = FIG_DIR / "port_turnaround_lookup_summary.svg"
    out_path.write_text("\n".join(svg_lines), encoding="utf-8")
    return out_path


def main() -> None:
    lookup, qa_bins = load_lookup_tables()
    out_path = write_combined_lookup_svg(lookup, qa_bins)
    print(f"Generated {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

