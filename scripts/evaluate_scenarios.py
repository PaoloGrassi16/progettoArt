from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from musicalai.evaluation.metrics import ScenarioRecord, summarize
from musicalai.scenarios.gaming_vr import build_plan as gaming_plan
from musicalai.scenarios.pro_audio import build_plan as pro_audio_plan
from musicalai.scenarios.therapy_masking import build_plan as therapy_plan


def _demo_records() -> list[ScenarioRecord]:
    """Create deterministic example records for checking the reporting pipeline."""
    rows: list[ScenarioRecord] = []
    gaming_values = [0.2, 0.45, 0.65, 0.8, 0.95]
    for index, intensity in enumerate(gaming_values, start=1):
        plan = gaming_plan(intensity=intensity, target="electronic")
        confidence = min(0.97, plan.confidence_threshold + 0.04 + intensity * 0.03)
        rows.append(
            ScenarioRecord(
                scenario="Gaming/VR",
                case_id=f"gaming-{index}",
                expected_genre=plan.target_genre,
                predicted_genre=plan.target_genre,
                confidence=round(confidence, 3),
                threshold=plan.confidence_threshold,
                approved=confidence >= plan.confidence_threshold,
                latency_ms=round(118 + intensity * 42, 2),
            )
        )

    novelty_values = [0.15, 0.35, 0.55, 0.75, 0.9]
    for index, novelty in enumerate(novelty_values, start=1):
        plan = pro_audio_plan(style="lofi_glitch", target="hiphop", novelty=novelty)
        confidence = min(0.96, plan.confidence_threshold + 0.08 - novelty * 0.02)
        rows.append(
            ScenarioRecord(
                scenario="Pro-Audio",
                case_id=f"proaudio-{index}",
                expected_genre=plan.target_genre,
                predicted_genre=plan.target_genre,
                confidence=round(confidence, 3),
                threshold=plan.confidence_threshold,
                approved=confidence >= plan.confidence_threshold,
                latency_ms=round(96 + novelty * 35, 2),
            )
        )

    noise_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    for index, noise in enumerate(noise_values, start=1):
        plan = therapy_plan(noise_level=noise, target="ambient")
        confidence = min(0.95, plan.confidence_threshold + 0.06 - noise * 0.025)
        rows.append(
            ScenarioRecord(
                scenario="Therapy/Masking",
                case_id=f"therapy-{index}",
                expected_genre=plan.target_genre,
                predicted_genre=plan.target_genre,
                confidence=round(confidence, 3),
                threshold=plan.confidence_threshold,
                approved=confidence >= plan.confidence_threshold,
                latency_ms=round(135 + noise * 55, 2),
            )
        )
    return rows


def _read_records(path: Path) -> list[ScenarioRecord]:
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                ScenarioRecord(
                    scenario=row["scenario"],
                    case_id=row["case_id"],
                    expected_genre=row["expected_genre"],
                    predicted_genre=row["predicted_genre"],
                    confidence=float(row["confidence"]),
                    threshold=float(row["threshold"]),
                    approved=row["approved"].strip().lower() in {"1", "true", "yes", "si", "sì"},
                    latency_ms=float(row["latency_ms"]),
                )
            )
    return records


def _write_records(path: Path, records: list[ScenarioRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ScenarioRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def _write_summary(path: Path, records: list[ScenarioRecord]) -> None:
    rows = summarize(records)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MusicalAI scenario results.")
    parser.add_argument("--input", type=Path, default=None, help="CSV with real predictions.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()

    records = _read_records(args.input) if args.input else _demo_records()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_records(args.output_dir / "scenario_results.csv", records)
    _write_summary(args.output_dir / "scenario_summary.csv", records)
    print(f"Written: {args.output_dir / 'scenario_results.csv'}")
    print(f"Written: {args.output_dir / 'scenario_summary.csv'}")


if __name__ == "__main__":
    main()
