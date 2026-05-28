"""Data-driven router tests.

This module loads ``data/command_router_examples.zh-en.yaml`` and asserts
that every utterance classifies as the YAML declares. The dataset is the
contract; the classifier patterns must evolve to satisfy it. See
``docs/sdd/command-router-acceptance.md`` for the full acceptance gate.

PyYAML is the parser. It is now declared in ``pyproject.toml`` (Track A
added it), but the test stays defensive with ``pytest.importorskip("yaml")``
so older dev environments without the dep just skip instead of erroring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from reachy_intent_runtime.classifier import RuleIntentClassifier  # noqa: E402

DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "command_router_examples.zh-en.yaml"
)


def _load_dataset() -> dict:
    with DATASET_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _flatten_examples() -> list[tuple[str, str, str, str | None, str | None, str]]:
    """Return ``[(group_name, utterance, exp_kind, exp_action, exp_priority, status), ...]``."""
    if not DATASET_PATH.exists():
        return []
    rows: list[tuple[str, str, str, str | None, str | None, str]] = []
    data = _load_dataset()
    for group in data.get("groups", []):
        expect = group.get("expect", {})
        status = group.get("status", "active")
        for utterance in group.get("examples", []):
            rows.append(
                (
                    group["name"],
                    utterance,
                    expect.get("kind"),
                    expect.get("action"),
                    expect.get("priority"),
                    status,
                )
            )
    return rows


@pytest.mark.parametrize(
    "group,utterance,exp_kind,exp_action,exp_priority,status",
    _flatten_examples(),
)
def test_router_example_classifies_as_expected(
    group: str,
    utterance: str,
    exp_kind: str,
    exp_action: str | None,
    exp_priority: str | None,
    status: str,
) -> None:
    """Every YAML example must classify as the YAML declares."""
    if status == "known_debt":
        pytest.xfail(f"[{group}] known classifier debt; pattern tightening will turn this red")
    result = RuleIntentClassifier().classify(utterance)
    assert result.kind == exp_kind, (
        f"[{group}] {utterance!r} → kind={result.kind!r}, expected {exp_kind!r}"
    )
    if exp_action is not None:
        assert result.action == exp_action, (
            f"[{group}] {utterance!r} → action={result.action!r}, expected {exp_action!r}"
        )
    if exp_priority is not None:
        # exp_priority is a string like "background"; classifier returns ActionPriority enum.
        assert result.priority is not None, (
            f"[{group}] {utterance!r} → priority=None, expected {exp_priority!r}"
        )
        assert result.priority.name.lower() == exp_priority, (
            f"[{group}] {utterance!r} → priority={result.priority.name.lower()!r}, "
            f"expected {exp_priority!r}"
        )


def test_dataset_has_minimum_coverage() -> None:
    rows = _flatten_examples()
    assert len(rows) >= 50, f"dataset has only {len(rows)} examples; minimum is 50"


def test_dataset_covers_all_intent_kinds() -> None:
    kinds = {row[2] for row in _flatten_examples()}
    assert kinds == {"command", "chat", "ambiguous"}, (
        f"dataset covers {kinds}; must cover {{'command', 'chat', 'ambiguous'}}"
    )
