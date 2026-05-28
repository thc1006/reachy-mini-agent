"""SDD-04 acceptance: extends CpuBudgetPolicy with CpuBudgetConfig parser/validator.

These tests are mock-only and run sub-second. They do not touch systemd or cgroups.
"""

from __future__ import annotations

import pytest

from reachy_intent_runtime.cpu_qos import (
    CpuBudgetConfig,
    parse_cpu_quota,
    parse_memory_max,
    validate_against_slice_cap,
    validate_budget_total,
)

# ---------------------------------------------------------------------------
# CpuBudgetConfig dataclass — basic construction
# ---------------------------------------------------------------------------


def test_cpu_budget_config_accepts_valid_fields() -> None:
    cfg = CpuBudgetConfig(
        cpu_weight=600,
        cpu_quota="80%",
        memory_max="384M",
        slice_name="reachy-runtime.slice",
        nice=-3,
    )
    assert cfg.cpu_weight == 600
    assert cfg.cpu_quota == "80%"
    assert cfg.memory_max == "384M"
    assert cfg.slice_name == "reachy-runtime.slice"
    assert cfg.nice == -3


def test_cpu_budget_config_rejects_cpu_weight_out_of_range() -> None:
    with pytest.raises(ValueError, match="cpu_weight"):
        CpuBudgetConfig(cpu_weight=0, cpu_quota="50%", memory_max="128M")
    with pytest.raises(ValueError, match="cpu_weight"):
        CpuBudgetConfig(cpu_weight=10001, cpu_quota="50%", memory_max="128M")


def test_cpu_budget_config_rejects_nice_out_of_range() -> None:
    with pytest.raises(ValueError, match="nice"):
        CpuBudgetConfig(cpu_weight=100, cpu_quota="50%", memory_max="128M", nice=-21)
    with pytest.raises(ValueError, match="nice"):
        CpuBudgetConfig(cpu_weight=100, cpu_quota="50%", memory_max="128M", nice=20)


# ---------------------------------------------------------------------------
# parse_cpu_quota — accepts systemd-style percentages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("80%", 0.80),
        ("120%", 1.20),
        ("40%", 0.40),
        ("100%", 1.00),
        ("400%", 4.00),  # 4 cores fully allocated
    ],
)
def test_parse_cpu_quota_valid(raw: str, expected: float) -> None:
    assert parse_cpu_quota(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "80", "abc%", "%", "-50%", "1.5"])
def test_parse_cpu_quota_invalid(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_cpu_quota(raw)


# ---------------------------------------------------------------------------
# parse_memory_max — accepts M / G suffixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected_bytes",
    [
        ("384M", 384 * 1024 * 1024),
        ("1G", 1 * 1024 * 1024 * 1024),
        ("2G", 2 * 1024 * 1024 * 1024),
        ("512M", 512 * 1024 * 1024),
        ("128M", 128 * 1024 * 1024),
    ],
)
def test_parse_memory_max_valid(raw: str, expected_bytes: int) -> None:
    assert parse_memory_max(raw) == expected_bytes


@pytest.mark.parametrize("raw", ["", "384", "abcM", "1T", "-1M", "1.5G"])
def test_parse_memory_max_invalid(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_memory_max(raw)


# ---------------------------------------------------------------------------
# validate_budget_total — sum of CPUQuota across services should fit Pi CM4
# ---------------------------------------------------------------------------


def test_validate_budget_total_warns_when_oversubscribed_beyond_cpu_count() -> None:
    # CM4 has 4 cores -> 400% absolute max; warn when oversubscribed > 1.5x.
    configs = [
        CpuBudgetConfig(cpu_weight=900, cpu_quota="80%", memory_max="64M"),
        CpuBudgetConfig(cpu_weight=600, cpu_quota="80%", memory_max="128M"),
        CpuBudgetConfig(cpu_weight=300, cpu_quota="120%", memory_max="384M"),
        CpuBudgetConfig(cpu_weight=150, cpu_quota="60%", memory_max="128M"),
        CpuBudgetConfig(cpu_weight=100, cpu_quota="400%", memory_max="64M"),
    ]
    warnings = validate_budget_total(configs, cpu_cores=4)
    # total quota = 0.8 + 0.8 + 1.2 + 0.6 + 4.0 = 7.4 (> 4 * 1.5 = 6.0 -> warn)
    assert any("oversubscribed" in w for w in warnings)


def test_validate_budget_total_no_warn_when_within_envelope() -> None:
    # Production split: 80 + 80 + 120 + 60 + 40 = 380% on a 4-core CM4
    configs = [
        CpuBudgetConfig(cpu_weight=900, cpu_quota="80%", memory_max="64M"),
        CpuBudgetConfig(cpu_weight=600, cpu_quota="80%", memory_max="128M"),
        CpuBudgetConfig(cpu_weight=300, cpu_quota="120%", memory_max="384M"),
        CpuBudgetConfig(cpu_weight=150, cpu_quota="60%", memory_max="128M"),
        CpuBudgetConfig(cpu_weight=100, cpu_quota="40%", memory_max="64M"),
    ]
    warnings = validate_budget_total(configs, cpu_cores=4)
    assert warnings == []


# ---------------------------------------------------------------------------
# validate_against_slice_cap — sum of per-service CPUQuota vs slice CPUQuota
# ---------------------------------------------------------------------------


def _production_configs() -> list[CpuBudgetConfig]:
    """The shipped 5-tier production split: 80 + 80 + 120 + 60 + 40 = 380%."""
    return [
        CpuBudgetConfig(cpu_weight=900, cpu_quota="80%", memory_max="64M"),
        CpuBudgetConfig(cpu_weight=600, cpu_quota="80%", memory_max="128M"),
        CpuBudgetConfig(cpu_weight=300, cpu_quota="120%", memory_max="384M"),
        CpuBudgetConfig(cpu_weight=150, cpu_quota="60%", memory_max="128M"),
        CpuBudgetConfig(cpu_weight=100, cpu_quota="40%", memory_max="64M"),
    ]


def test_validate_against_slice_cap_no_warning_under_cap() -> None:
    """Total 380% under a 400% slice cap -> no warning."""
    warnings = validate_against_slice_cap(_production_configs(), slice_cap_pct=400.0)
    assert warnings == []


def test_validate_against_slice_cap_no_warning_at_cap_exact_match() -> None:
    """Total 380% exactly equals the shipped slice CPUQuota=380% -> no warning."""
    warnings = validate_against_slice_cap(_production_configs(), slice_cap_pct=380.0)
    assert warnings == []


def test_validate_against_slice_cap_warns_when_over_cap() -> None:
    """Adding a 200% greedy worker pushes total to 580% > 380% cap -> warning."""
    configs = _production_configs() + [
        CpuBudgetConfig(cpu_weight=100, cpu_quota="200%", memory_max="64M"),
    ]
    warnings = validate_against_slice_cap(configs, slice_cap_pct=380.0)
    assert any("slice_cap_exceeded" in w for w in warnings)
    # Warning must include both numbers so operators can see the overshoot.
    assert any("580" in w and "380" in w for w in warnings)


def test_validate_against_slice_cap_defaults_to_380_pct() -> None:
    """The default cap matches the shipped reachy-runtime.slice CPUQuota=380%."""
    # 5 configs each at 100% = 500% > 380% -> must warn under default.
    configs = [
        CpuBudgetConfig(cpu_weight=100, cpu_quota="100%", memory_max="64M") for _ in range(5)
    ]
    warnings = validate_against_slice_cap(configs)
    assert any("slice_cap_exceeded" in w for w in warnings)


def test_validate_against_slice_cap_empty_list_no_warning() -> None:
    assert validate_against_slice_cap([], slice_cap_pct=380.0) == []


# ---------------------------------------------------------------------------
# Existing CpuBudgetPolicy still works (regression guard)
# ---------------------------------------------------------------------------


def test_existing_cpu_budget_policy_still_works() -> None:
    from reachy_intent_runtime.resource_policy import CpuBudgetPolicy

    warnings = CpuBudgetPolicy(max_cpu_percent=70).evaluate(cpu_percent=80, missed_heartbeats=0)
    assert warnings == ["cpu_over_budget:80.0>70.0"]
