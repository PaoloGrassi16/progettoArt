from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from musicalai.config import ClosedLoopConfig, GanConfig, MusicalAIConfig, PerceptionConfig
from musicalai.scenarios.common import map_to_gtzan


def cmd_prepare_data(args: argparse.Namespace) -> None:
    from musicalai.data.prepare import prepare_feature_dataset

    counts = prepare_feature_dataset(args.audio_root, args.output, MusicalAIConfig().audio)
    for genre, count in sorted(counts.items()):
        print(f"{genre}: {count} files")


def cmd_train_classifier(args: argparse.Namespace) -> None:
    from musicalai.training.train_classifier import train_classifier

    checkpoint = train_classifier(
        feature_root=args.feature_root,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    print(f"Saved classifier checkpoint: {checkpoint}")


def cmd_train_gan(args: argparse.Namespace) -> None:
    from musicalai.training.train_gan import train_gan

    checkpoint = train_gan(
        feature_root=args.feature_root,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    print(f"Saved GAN checkpoint: {checkpoint}")


def cmd_generate(args: argparse.Namespace) -> None:
    import torch

    from musicalai.generation.closed_loop import (
        ClosedLoopSoundscapeGenerator,
        save_generation_outputs,
    )
    from musicalai.models.dcgan import SpectrogramGenerator
    from musicalai.models.perception import CNNBiGRUAttentionClassifier

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    classifier_payload = torch.load(args.classifier_checkpoint, map_location=device)
    generator_payload = torch.load(args.generator_checkpoint, map_location=device)

    classes = list(classifier_payload["classes"])
    perception_config = PerceptionConfig(**classifier_payload["config"])
    gan_config = GanConfig(**generator_payload["config"])
    validator = CNNBiGRUAttentionClassifier(perception_config).to(device)
    validator.load_state_dict(classifier_payload["model_state"])
    for param in validator.parameters():
        param.requires_grad_(False)

    generator = SpectrogramGenerator(gan_config).to(device)
    generator.load_state_dict(generator_payload["generator_state"])

    loop_config = ClosedLoopConfig(
        confidence_threshold=args.threshold,
        max_attempts=args.max_attempts,
        device=str(device),
    )
    target = map_to_gtzan(args.target)
    loop = ClosedLoopSoundscapeGenerator(
        generator=generator,
        validator=validator,
        classes=classes,
        gan_config=gan_config,
        loop_config=loop_config,
    )
    result = loop.generate(target)
    paths = save_generation_outputs(result, args.output, MusicalAIConfig().audio)
    print(
        "approved={approved} target={target} predicted={predicted} "
        "confidence={confidence:.3f} attempts={attempts}".format(**result.__dict__)
    )
    for kind, path in paths.items():
        print(f"{kind}: {path}")


def cmd_scenario_gaming(args: argparse.Namespace) -> None:
    from musicalai.scenarios.gaming_vr import build_plan

    print(build_plan(args.intensity, args.target, args.environment).to_json())


def cmd_scenario_pro_audio(args: argparse.Namespace) -> None:
    from musicalai.scenarios.pro_audio import build_plan

    print(build_plan(args.style, args.target, args.novelty).to_json())


def cmd_scenario_therapy(args: argparse.Namespace) -> None:
    from musicalai.scenarios.therapy_masking import build_plan

    print(build_plan(args.noise_level, args.target, args.setting).to_json())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="musicalai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--audio-root", required=True)
    prepare.add_argument("--output", default="data/features")
    prepare.set_defaults(func=cmd_prepare_data)

    classifier = subparsers.add_parser("train-classifier")
    classifier.add_argument("--feature-root", required=True)
    classifier.add_argument("--output", default="runs/perception")
    classifier.add_argument("--epochs", type=int, default=25)
    classifier.add_argument("--batch-size", type=int, default=16)
    classifier.add_argument("--learning-rate", type=float, default=1e-4)
    classifier.add_argument("--device", default=None)
    classifier.set_defaults(func=cmd_train_classifier)

    gan = subparsers.add_parser("train-gan")
    gan.add_argument("--feature-root", required=True)
    gan.add_argument("--output", default="runs/dcgan")
    gan.add_argument("--epochs", type=int, default=50)
    gan.add_argument("--batch-size", type=int, default=32)
    gan.add_argument("--learning-rate", type=float, default=2e-4)
    gan.add_argument("--device", default=None)
    gan.set_defaults(func=cmd_train_gan)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--target", required=True)
    generate.add_argument("--classifier-checkpoint", required=True)
    generate.add_argument("--generator-checkpoint", required=True)
    generate.add_argument("--output", default="outputs/generated")
    generate.add_argument("--threshold", type=float, default=0.85)
    generate.add_argument("--max-attempts", type=int, default=64)
    generate.add_argument("--device", default=None)
    generate.set_defaults(func=cmd_generate)

    gaming = subparsers.add_parser("scenario-gaming-vr")
    gaming.add_argument("--intensity", type=float, default=0.75)
    gaming.add_argument("--target", default="electronic")
    gaming.add_argument("--environment", default="vr_arena")
    gaming.set_defaults(func=cmd_scenario_gaming)

    pro_audio = subparsers.add_parser("scenario-pro-audio")
    pro_audio.add_argument("--style", default="lofi_glitch")
    pro_audio.add_argument("--target", default="hiphop")
    pro_audio.add_argument("--novelty", type=float, default=0.7)
    pro_audio.set_defaults(func=cmd_scenario_pro_audio)

    therapy = subparsers.add_parser("scenario-therapy")
    therapy.add_argument("--noise-level", type=float, default=0.65)
    therapy.add_argument("--target", default="ambient")
    therapy.add_argument("--setting", default="urban_waiting_room")
    therapy.set_defaults(func=cmd_scenario_therapy)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

