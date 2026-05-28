from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classifier import RuleIntentClassifier
from .models import MotionCommand
from .motion_adapter import MockMotionAdapter
from .scheduler import ActionScheduler


def command_from_action(action: str, utterance: str, priority_name: str) -> MotionCommand:
    priority = {"critical": 0, "interactive": 10, "background": 20}[priority_name]
    from .models import ActionPriority

    return MotionCommand(
        name=action,
        tool=action,
        priority=ActionPriority(priority),
        interruptible=priority_name != "critical",
        duration_ms=30000 if action == "dance" else 500,
        chunk_ms=500 if action == "dance" else 100,
        metadata={"utterance": utterance},
    )


def run_script(path: Path) -> list[str]:
    classifier = RuleIntentClassifier()
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    rows = json.loads(path.read_text(encoding="utf-8"))
    output: list[str] = []
    for row in rows:
        elapsed = int(row.get("elapsed_ms", 0))
        if elapsed:
            scheduler.tick(elapsed)
        utterance = row.get("utterance")
        if utterance:
            result = classifier.classify(utterance)
            output.append(
                f"utterance={utterance!r} -> {result.kind}:{result.action}:{result.priority}"
            )
            if result.kind == "command" and result.action and result.priority is not None:
                cmd = command_from_action(result.action, utterance, result.priority.name.lower())
                scheduler.submit(cmd)
    output.extend(f"adapter:{item}" for item in adapter.log)
    output.extend(
        f"event:{event.event}:{event.command}:{event.details}" for event in scheduler.events
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    args = parser.parse_args()
    for line in run_script(args.script):
        print(line)


if __name__ == "__main__":
    main()
