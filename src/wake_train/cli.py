"""CLI for wake-word training.

Subcommands run pipeline stages independently; ``all`` chains them.

Examples:
    PYTHONPATH=src python -m wake_train.cli all --scope poc
    PYTHONPATH=src python -m wake_train.cli synth --scope poc
    PYTHONPATH=src python -m wake_train.cli eval --scope poc
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import config as cfgmod
from . import manifest as mfst
from . import negatives as neg
from . import phrases as ph


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )


def cmd_synth(args: argparse.Namespace) -> int:
    from .synth import synthesize_phrase_set

    cfg = cfgmod.get(args.scope)
    phrase_objs = tuple(ph.resolve(cfg.phrases))
    cfg.synth_dir.mkdir(parents=True, exist_ok=True)
    synthesize_phrase_set(
        voices=cfg.voices,
        phrases=phrase_objs,
        out_dir=cfg.synth_dir,
        n_per_voice_phrase=cfg.n_synth_per_voice_phrase,
        sample_rate=cfg.train.sample_rate,
    )
    return 0


def cmd_augment(args: argparse.Namespace) -> int:
    """Walk synth dir, apply augmentation pipeline, write to aug dir.

    Each (clip, copy_idx) gets a deterministic seed derived from the path +
    copy index, so inserting a new clip into the synth dir does NOT shift
    the augmentation of any prior clip.
    """
    import hashlib

    import soundfile as sf

    from .augment import AugPipeline

    cfg = cfgmod.get(args.scope)
    pipeline = AugPipeline(cfg.augment, sample_rate=cfg.train.sample_rate)
    cfg.aug_dir.mkdir(parents=True, exist_ok=True)
    base_seed = cfg.augment.seed
    n = 0
    for wav in sorted(cfg.synth_dir.rglob("*.wav")):
        rel = wav.relative_to(cfg.synth_dir)
        out = cfg.aug_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        for k in range(args.n_per_clip):
            key = f"{rel.as_posix()}|{k}|{base_seed}".encode()
            clip_seed = int.from_bytes(hashlib.sha1(key).digest()[:4], "big")
            mutated = pipeline.apply(audio, clip_seed=clip_seed)
            stem, suffix = out.stem, out.suffix
            sf.write(str(out.with_name(f"{stem}_aug{k:02d}{suffix}")),
                     mutated, sr, subtype="PCM_16")
            n += 1
    print(f"augmented clips written: {n}")
    return 0


def cmd_negatives(args: argparse.Namespace) -> int:
    cfg = cfgmod.get(args.scope)
    neg.fetch(cfg.neg_dir, filenames=cfg.negative_files)
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    cfg = cfgmod.get(args.scope)
    mfst.build(cfg)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train import train

    cfg = cfgmod.get(args.scope)
    train(cfg)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .export import export

    cfg = cfgmod.get(args.scope)
    export(cfg)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval import measure_latency, measure_accuracy

    cfg = cfgmod.get(args.scope)
    lat = measure_latency(cfg)
    acc = measure_accuracy(cfg)
    report = {"latency": lat, "accuracy": acc}
    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    for sub in ("negatives", "synth", "augment", "manifest", "train", "export", "eval"):
        rc = COMMANDS[sub](args)
        if rc != 0:
            return rc
    return 0


COMMANDS = {
    "synth": cmd_synth,
    "augment": cmd_augment,
    "negatives": cmd_negatives,
    "manifest": cmd_manifest,
    "train": cmd_train,
    "export": cmd_export,
    "eval": cmd_eval,
    "all": cmd_all,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wake_train")
    p.add_argument("--verbose", "-v", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in COMMANDS:
        sp = sub.add_parser(name)
        sp.add_argument("--scope", choices=["poc", "prod"], default="poc")
        if name in ("augment", "all"):
            sp.add_argument("--n-per-clip", type=int, default=2,
                            help="augmented copies per synth clip")
    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    return COMMANDS[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
