from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLORS = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c"]


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["scenario"] != "overall"]


def _bar_chart(
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    output: Path,
    y_label: str,
    percent: bool = True,
) -> None:
    width = 920
    height = 560
    margin_left = 90
    margin_bottom = 90
    plot_w = width - margin_left - 50
    plot_h = height - 120
    max_value = 1.0 if percent else max(float(row[metric]) for row in rows) * 1.15
    bar_gap = 36
    bar_w = (plot_w - bar_gap * (len(rows) - 1)) / len(rows)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="45" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">{title}</text>',
        f'<text x="26" y="{height / 2}" transform="rotate(-90 26 {height / 2})" text-anchor="middle" font-family="Arial" font-size="15">{y_label}</text>',
    ]

    # grid and y-axis labels
    for tick in range(0, 6):
        value = max_value * tick / 5
        y = 70 + plot_h - (value / max_value) * plot_h
        label = f"{value * 100:.0f}%" if percent else f"{value:.0f}"
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - 50}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#374151">{label}</text>')

    parts.append(f'<line x1="{margin_left}" y1="70" x2="{margin_left}" y2="{70 + plot_h}" stroke="#111827"/>')
    parts.append(f'<line x1="{margin_left}" y1="{70 + plot_h}" x2="{width - 50}" y2="{70 + plot_h}" stroke="#111827"/>')

    for index, row in enumerate(rows):
        value = float(row[metric])
        x = margin_left + index * (bar_w + bar_gap)
        bar_h = (value / max_value) * plot_h
        y = 70 + plot_h - bar_h
        color = COLORS[index % len(COLORS)]
        label = f"{value * 100:.1f}%" if percent else f"{value:.1f}"
        scenario = row["scenario"]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="4"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700">{label}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - 52}" text-anchor="middle" font-family="Arial" font-size="14">{scenario}</text>')

    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SVG charts from scenario_summary.csv.")
    parser.add_argument("--summary", type=Path, default=Path("outputs/evaluation/scenario_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation/figures"))
    args = parser.parse_args()
    rows = _read_summary(args.summary)
    _bar_chart(rows, "accuracy", "Accuracy per scenario", args.output_dir / "accuracy_by_scenario.svg", "Accuracy", percent=True)
    _bar_chart(rows, "approval_rate", "Tasso di approvazione closed-loop", args.output_dir / "approval_rate_by_scenario.svg", "Approvazione", percent=True)
    _bar_chart(rows, "avg_confidence", "Confidenza media del validatore", args.output_dir / "confidence_by_scenario.svg", "Confidenza", percent=True)
    _bar_chart(rows, "avg_latency_ms", "Latenza media per scenario", args.output_dir / "latency_by_scenario.svg", "Millisecondi", percent=False)
    print(f"Written charts to: {args.output_dir}")


if __name__ == "__main__":
    main()

