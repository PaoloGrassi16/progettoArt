from __future__ import annotations

import argparse

from musicalai.scenarios.common import ScenarioPlan, clamp, map_to_gtzan


def build_plan(
    noise_level: float,
    target: str = "ambient",
    setting: str = "urban_waiting_room",
) -> ScenarioPlan:
    noise_level = clamp(noise_level)
    masking_strength = 0.4 + noise_level * 0.5
    threshold = 0.88
    mapped = map_to_gtzan(target)
    return ScenarioPlan(
        name="Mascheramento acustico e musicoterapia ambientale",
        target_genre=mapped,
        duration_seconds=30.0,
        confidence_threshold=threshold,
        prompt=(
            "Genera un soundscape rilassante che attenui picchi casuali "
            "di rumore ambientale e mantenga bassa densita' ritmica."
        ),
        controls={
            "setting": setting,
            "target_style": target,
            "noise_level": round(noise_level, 2),
            "masking_strength": round(masking_strength, 2),
            "low_frequency_softening": round(0.25 + noise_level * 0.35, 2),
            "avoid_transients": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-level", type=float, default=0.65)
    parser.add_argument("--target", default="ambient")
    parser.add_argument("--setting", default="urban_waiting_room")
    args = parser.parse_args()
    print(build_plan(args.noise_level, args.target, args.setting).to_json())


if __name__ == "__main__":
    main()

