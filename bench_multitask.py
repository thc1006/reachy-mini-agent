"""B2 benchmark: serial baseline vs concurrent target architecture.

Runs the same synthetic conversation through both runners, ``N`` turns
each, prints a side-by-side latency comparison, and optionally writes
per-turn timings to ``bench_data/<timestamp>/bench_multitask.json``.

Defaults are calibrated from production memory (see
``docs/multitask-arch.md`` §5). Override via CLI flags for sweeps.

Usage::

    python bench_multitask.py
    python bench_multitask.py --turns 30 --json out.json
    python bench_multitask.py --quick

The bench is fully self-contained: no robot hardware, no GPU, no
network. Each run takes ~``turns × turn_period_s`` seconds.

What the bench is *not*: it does not run real STT/LLM/TTS. It runs
the orchestrator on synthetic timed workloads so the *architectural*
delta is measurable in isolation. The real-hardware bench lives in
``robot_brain.py`` and is a follow-up PR (see migration plan).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from orchestrator import (  # noqa: E402
    ConcurrentRunner,
    DialogConfig,
    MotionConfig,
    PerceptionConfig,
    RunResult,
    SceneDescribed,
    SerialRunner,
    TurnTiming,
)


UTTERANCES = (
    ("Hi Reachy, can you wave at me please?", 1200.0),
    ("What do you see in front of you?", 1100.0),
    ("Look to the right and tell me what is there.", 1400.0),
    ("Could you nod if you understand me?", 1300.0),
    ("Tell me a short joke about robots.", 1200.0),
    ("How long did it take you to answer that?", 1500.0),
    ("Demo a happy face for me.", 1000.0),
    ("Can you describe the room briefly?", 1400.0),
    ("Try a tracking move toward the camera.", 1600.0),
    ("OK that is enough, please relax now.", 1300.0),
)


@dataclass
class Pctl:
    p50: float
    p95: float
    mean: float
    min: float
    max: float


def pctl(values: List[float]) -> Pctl:
    if not values:
        return Pctl(0, 0, 0, 0, 0)
    vs = sorted(values)
    n = len(vs)
    def q(p: float) -> float:
        # nearest-rank (small N — keep it simple, no interpolation)
        k = max(0, min(n - 1, int(round(p * (n - 1)))))
        return vs[k]
    return Pctl(
        p50=q(0.5),
        p95=q(0.95),
        mean=statistics.fmean(vs),
        min=vs[0],
        max=vs[-1],
    )


def build_configs(stt_endpointing_ms: float = 150.0,
                  periodic_vision_interval_ms: float = 30000.0):
    """Build the three config dataclasses. Two perception configs are
    returned so serial vs concurrent get their *appropriate* vision
    policy without leaking the architectural difference into the bench
    runner.

    Returns ``(perception_serial, perception_concurrent, dialog, motion)``.
    """
    common = dict(
        silence_wait_ms=1400.0,
        stt_batch_ms=200.0,
        stt_partial_cadence_ms=400.0,
        stt_final_after_user_stop_ms=stt_endpointing_ms,
        vision_caption_ms=1000.0,
        mic_drain_after_speak_ms=400.0,
    )
    # Serial: no on-demand vision; periodic 30s tick mirrors production
    # ``vision_worker``.
    perception_serial = PerceptionConfig(
        **common,
        enable_on_demand_vision=False,
        periodic_vision_interval_ms=periodic_vision_interval_ms,
    )
    # Concurrent: on-demand vision via ``dialog.thinking`` event; no
    # periodic tick (the on-demand path is sufficient).
    perception_concurrent = PerceptionConfig(
        **common,
        enable_on_demand_vision=True,
        periodic_vision_interval_ms=0.0,
    )
    dialog = DialogConfig(
        llm_ttfb_ms=350.0,
        llm_chunk_count=4,
        llm_chunk_gen_ms=120.0,
        tts_synth_ms_per_chunk=300.0,
        tts_play_ms_per_chunk=350.0,
        tts_max_concurrent=2,
        tool_at_chunk=1,
        use_partial_prefill=True,
    )
    motion = MotionConfig()
    return perception_serial, perception_concurrent, dialog, motion


def prewarm_scene(runner) -> None:
    """Publish one synthetic scene so the *first* turn's scene_age
    is finite. Without this, turn 0 always shows scene_age=inf in
    both paths and the comparison is uninformative for turn 0."""
    runner.bus.publish(SceneDescribed(text="(prewarm) the user faces the camera"))
    # ensure subscriber drained before we start the turn
    time.sleep(0.02)


def run(label: str, runner, n_turns: int, turn_period_s: float) -> RunResult:
    print(f"\n=== {label} — {n_turns} turns ===")
    runner.start()
    prewarm_scene(runner)
    timings: List[TurnTiming] = []
    try:
        for i in range(n_turns):
            text, dur_ms = UTTERANCES[i % len(UTTERANCES)]
            t = runner.run_turn(text, dur_ms)
            timings.append(t)
            print(
                f"  turn {i + 1:2d}: ttfb={t.ttfb_audio_ms:6.0f}ms "
                f"total={t.turn_total_ms:6.0f}ms "
                f"mic_block={t.mic_blocked_ms:5.0f}ms "
                f"scene_age={_fmt_age(t.scene_age_at_llm_ms):>8s} "
                f"warm={int(t.prefill_warm)}"
            )
            if i + 1 < n_turns:
                time.sleep(turn_period_s)
    finally:
        runner.stop()
    return runner.snapshot(label, timings)


def _fmt_age(ms: float) -> str:
    if ms == float("inf"):
        return "inf"
    return f"{ms:.0f}ms"


def summarise(label: str, result: RunResult) -> dict:
    ttfb = pctl([t.ttfb_audio_ms for t in result.timings if t.ttfb_audio_ms != float("inf")])
    total = pctl([t.turn_total_ms for t in result.timings if t.turn_total_ms != float("inf")])
    mic   = pctl([t.mic_blocked_ms for t in result.timings])
    # scene_age can be inf on first turn — separate that out so percentiles
    # remain meaningful.
    finite_ages = [t.scene_age_at_llm_ms for t in result.timings if t.scene_age_at_llm_ms != float("inf")]
    age = pctl(finite_ages)
    n_inf_age = sum(1 for t in result.timings if t.scene_age_at_llm_ms == float("inf"))
    return {
        "label": label,
        "n": len(result.timings),
        "ttfb_audio_ms": asdict(ttfb),
        "turn_total_ms": asdict(total),
        "mic_blocked_ms": asdict(mic),
        "scene_age_finite_ms": asdict(age),
        "scene_age_inf_count": n_inf_age,
        "bus_stats": result.bus_stats,
        "perception": {
            "partials_emitted": result.perception_stats.partials_emitted,
            "partials_dropped_during_speak": result.perception_stats.partials_dropped_during_speak,
            "mic_blocked_ms_total": result.perception_stats.mic_blocked_ms_total,
            "speak_windows_count": len(result.perception_stats.speak_windows),
        },
        "dialog_n_turns": result.dialog_stats.n_turns,
        "motion": {
            "n_tracking_setpoints": result.motion_stats.n_tracking_setpoints,
            "n_tool_calls": result.motion_stats.n_tool_calls,
            "n_recenters": result.motion_stats.n_recenters,
        },
    }


def print_table(serial_sum: dict, concurrent_sum: dict) -> None:
    def fmt(d: dict, key: str, pct: str) -> str:
        return f"{d[key][pct]:.0f}"

    def delta(s: dict, c: dict, key: str, pct: str) -> str:
        sv = s[key][pct]
        cv = c[key][pct]
        if sv == 0:
            return "  n/a"
        pct_delta = (cv - sv) / sv * 100
        sign = "" if pct_delta < 0 else "+"
        return f"{sign}{pct_delta:5.1f}%"

    print("\n=== Summary ===")
    print(f"{'metric':30s} {'serial':>10s} {'concurrent':>12s} {'delta':>8s}")
    print("-" * 64)
    for metric in ("ttfb_audio_ms", "turn_total_ms", "mic_blocked_ms",
                   "scene_age_finite_ms"):
        for pct in ("p50", "p95"):
            row = f"{metric:25s} {pct:>4s}"
            row += f"  {fmt(serial_sum, metric, pct):>8s}ms"
            row += f"   {fmt(concurrent_sum, metric, pct):>8s}ms"
            row += f"   {delta(serial_sum, concurrent_sum, metric, pct):>7s}"
            print(row)
    print()
    print(f"scene_age=inf turns:  serial={serial_sum['scene_age_inf_count']}"
          f"  concurrent={concurrent_sum['scene_age_inf_count']}")
    print(f"perception.partials_emitted:  serial={serial_sum['perception']['partials_emitted']}"
          f"  concurrent={concurrent_sum['perception']['partials_emitted']}")
    print(f"perception.mic_blocked_ms_total: serial={serial_sum['perception']['mic_blocked_ms_total']:.0f}"
          f"  concurrent={concurrent_sum['perception']['mic_blocked_ms_total']:.0f}")
    print(f"motion.tool_calls: serial={serial_sum['motion']['n_tool_calls']}"
          f"  concurrent={concurrent_sum['motion']['n_tool_calls']}")
    print(f"motion.tracking_setpoints: serial={serial_sum['motion']['n_tracking_setpoints']}"
          f"  concurrent={concurrent_sum['motion']['n_tracking_setpoints']}")
    # Sanity check the bus didn't silently drop or block anything we
    # didn't *expect* to drop (face.seen with DROP_OLDEST is fine).
    for label, sum_ in (("serial", serial_sum), ("concurrent", concurrent_sum)):
        for sub in sum_["bus_stats"]:
            if sub["dropped"] and "face" not in sub["name"]:
                print(f"  [bus warn] {label}/{sub['name']} dropped={sub['dropped']}")
            if sub["blocked_publishes"]:
                print(f"  [bus warn] {label}/{sub['name']} "
                      f"blocked_publishes={sub['blocked_publishes']}")


def check_pass_criteria(serial_sum: dict, conc_sum: dict) -> int:
    """Returns exit code: 0 if all targets met, 1 otherwise. Targets
    are documented in ``docs/multitask-arch.md`` §5."""
    s_ttfb50 = serial_sum["ttfb_audio_ms"]["p50"]
    c_ttfb50 = conc_sum["ttfb_audio_ms"]["p50"]
    s_total50 = serial_sum["turn_total_ms"]["p50"]
    c_total50 = conc_sum["turn_total_ms"]["p50"]
    c_mic50 = conc_sum["mic_blocked_ms"]["p50"]

    criteria = [
        ("ttfb_audio_ms p50 >= 30% reduction",
         c_ttfb50 <= s_ttfb50 * 0.70,
         f"serial={s_ttfb50:.0f} concurrent={c_ttfb50:.0f}"),
        ("turn_total_ms p50 >= 15% reduction",
         c_total50 <= s_total50 * 0.85,
         f"serial={s_total50:.0f} concurrent={c_total50:.0f}"),
        ("mic_blocked_ms p50 == 0",
         c_mic50 == 0.0,
         f"concurrent={c_mic50:.0f}"),
    ]
    print("\n=== Pass criteria ===")
    all_ok = True
    for name, ok, detail in criteria:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}  ({detail})")
        all_ok = all_ok and ok
    return 0 if all_ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--turns", type=int, default=20,
                   help="number of turns per scenario (default 20)")
    p.add_argument("--turn-period-s", type=float, default=1.5,
                   help="real-time gap between turns (default 1.5s)")
    p.add_argument("--json", type=Path, default=None,
                   help="optional path to write detailed JSON output")
    p.add_argument("--quick", action="store_true",
                   help="quick mode: turns=5, gap=0.5s (for dev iteration)")
    p.add_argument("--stt-endpointing-ms", type=float, default=150.0,
                   help="streaming-STT final-after-user-stop latency. "
                        "150ms is optimistic (WhisperLiveKit best-case), "
                        "350ms is realistic.")
    p.add_argument("--periodic-vision-interval-ms", type=float, default=30000.0,
                   help="serial-baseline periodic vision tick in ms "
                        "(default 30000 = production setting)")
    args = p.parse_args(argv)

    n = args.turns
    gap = args.turn_period_s
    if args.quick:
        n, gap = 5, 0.5

    perception_serial, perception_concurrent, dialog_cfg, motion_cfg = build_configs(
        stt_endpointing_ms=args.stt_endpointing_ms,
        periodic_vision_interval_ms=args.periodic_vision_interval_ms,
    )

    serial = SerialRunner(
        perception_cfg=perception_serial,
        dialog_cfg=DialogConfig(**{**dialog_cfg.__dict__, "use_partial_prefill": False}),
        motion_cfg=motion_cfg,
    )
    s_result = run("Serial (baseline)", serial, n, gap)

    conc = ConcurrentRunner(
        perception_cfg=perception_concurrent,
        dialog_cfg=dialog_cfg,
        motion_cfg=motion_cfg,
        streaming_stt=True,
        gate_mic_during_speak=False,
    )
    c_result = run("Concurrent (target)", conc, n, gap)

    s_sum = summarise("serial", s_result)
    c_sum = summarise("concurrent", c_result)

    print_table(s_sum, c_sum)
    rc = check_pass_criteria(s_sum, c_sum)

    if args.json is not None:
        out = {
            "turns": n,
            "turn_period_s": gap,
            "stt_endpointing_ms": args.stt_endpointing_ms,
            "periodic_vision_interval_ms": args.periodic_vision_interval_ms,
            "perception_cfg_serial": perception_serial.__dict__,
            "perception_cfg_concurrent": perception_concurrent.__dict__,
            "dialog_cfg": dialog_cfg.__dict__,
            "motion_cfg": motion_cfg.__dict__,
            "serial": {
                "summary": s_sum,
                "turns": [t.__dict__ for t in s_result.timings],
            },
            "concurrent": {
                "summary": c_sum,
                "turns": [t.__dict__ for t in c_result.timings],
            },
            "ts": time.time(),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=_json_default)
        print(f"\nDetailed JSON written to {args.json}")

    return rc


def _json_default(o):
    if o == float("inf"):
        return "inf"
    raise TypeError(f"not serialisable: {type(o)!r}")


if __name__ == "__main__":
    sys.exit(main())
