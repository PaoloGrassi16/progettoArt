from __future__ import annotations

import argparse

from musicalai.scenarios.common import ScenarioPlan, clamp, map_to_gtzan


def build_plan(
    intensity: float,
    target: str = "electronic",
    environment: str = "vr_arena",
) -> ScenarioPlan:
    intensity = clamp(intensity)
    density = 0.35 + intensity * 0.6
    tempo = 72 + int(intensity * 78)
    threshold = 0.78 + intensity * 0.12
    mapped = map_to_gtzan(target)
    return ScenarioPlan(
        name="Soundtrack adattiva per media interattivi",
        target_genre=mapped,
        duration_seconds=12.0,
        confidence_threshold=round(threshold, 2),
        prompt=(
            "Genera un loop dinamico coerente con lo stato di gioco, "
            "senza materiale coperto da copyright."
        ),
        controls={
            "environment": environment,
            "interaction_intensity": round(intensity, 2),
            "target_style": target,
            "rhythmic_density": round(density, 2),
            "suggested_bpm": tempo,
            "seamless_loop": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intensity", type=float, default=0.75)
    parser.add_argument("--target", default="electronic")
    parser.add_argument("--environment", default="vr_arena")
    args = parser.parse_args()
    print(build_plan(args.intensity, args.target, args.environment).to_json())


if __name__ == "__main__":
    main()

