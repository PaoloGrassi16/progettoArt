from __future__ import annotations

import argparse

from musicalai.scenarios.common import ScenarioPlan, clamp, map_to_gtzan


def build_plan(
    style: str = "lofi_glitch",
    target: str = "hiphop",
    novelty: float = 0.7,
) -> ScenarioPlan:
    novelty = clamp(novelty)
    mapped = map_to_gtzan(style if style else target)
    return ScenarioPlan(
        name="Co-pilota creativo per produzione musicale",
        target_genre=mapped,
        duration_seconds=8.0,
        confidence_threshold=0.82,
        prompt=(
            "Genera texture e sample inediti con imperfezioni controllate "
            "da estetica Lo-Fi/Glitch Art."
        ),
        controls={
            "requested_style": style,
            "producer_target": target,
            "novelty": round(novelty, 2),
            "glitch_amount": round(0.25 + novelty * 0.55, 2),
            "warmth": round(0.85 - novelty * 0.25, 2),
            "export_stems": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", default="lofi_glitch")
    parser.add_argument("--target", default="hiphop")
    parser.add_argument("--novelty", type=float, default=0.7)
    args = parser.parse_args()
    print(build_plan(args.style, args.target, args.novelty).to_json())


if __name__ == "__main__":
    main()

