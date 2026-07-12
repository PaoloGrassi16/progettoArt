from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class ScenarioRecord:
    scenario: str
    case_id: str
    expected_genre: str
    predicted_genre: str
    confidence: float
    threshold: float
    approved: bool
    latency_ms: float

    @property
    def correct(self) -> bool:
        return self.expected_genre == self.predicted_genre


def accuracy(records: list[ScenarioRecord]) -> float:
    if not records:
        return 0.0
    return sum(record.correct for record in records) / len(records)


def approval_rate(records: list[ScenarioRecord]) -> float:
    if not records:
        return 0.0
    return sum(record.approved for record in records) / len(records)


def group_by_scenario(records: list[ScenarioRecord]) -> dict[str, list[ScenarioRecord]]:
    grouped: dict[str, list[ScenarioRecord]] = defaultdict(list)
    for record in records:
        grouped[record.scenario].append(record)
    return dict(grouped)


def summarize(records: list[ScenarioRecord]) -> list[dict[str, float | str | int]]:
    rows = []
    for scenario, items in sorted(group_by_scenario(records).items()):
        rows.append(
            {
                "scenario": scenario,
                "n": len(items),
                "accuracy": round(accuracy(items), 4),
                "approval_rate": round(approval_rate(items), 4),
                "avg_confidence": round(mean(item.confidence for item in items), 4),
                "avg_latency_ms": round(mean(item.latency_ms for item in items), 2),
            }
        )
    rows.append(
        {
            "scenario": "overall",
            "n": len(records),
            "accuracy": round(accuracy(records), 4),
            "approval_rate": round(approval_rate(records), 4),
            "avg_confidence": round(mean(item.confidence for item in records), 4) if records else 0.0,
            "avg_latency_ms": round(mean(item.latency_ms for item in records), 2) if records else 0.0,
        }
    )
    return rows

