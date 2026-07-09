from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScenarioPlan:
    name: str
    target_genre: str
    duration_seconds: float
    confidence_threshold: float
    prompt: str
    controls: dict[str, float | str | bool]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def map_to_gtzan(target: str) -> str:
    mapping = {
        "ambient": "classical",
        "electronic": "disco",
        "lofi": "hiphop",
        "lofi_glitch": "hiphop",
        "vr": "disco",
        "calm": "classical",
        "stress_relief": "classical",
    }
    return mapping.get(target, target)

