"""SDD-04 helpers: parse systemd CPU/memory budget strings and validate totals.

Pure helpers — no systemd / cgroup calls. Importable on any platform.
"""

from __future__ import annotations

import configparser
import re
from dataclasses import dataclass
from pathlib import Path

_CPU_QUOTA_RE = re.compile(r"^(\d+)%$")
_MEMORY_MAX_RE = re.compile(r"^(\d+)([MG])$")


@dataclass(frozen=True)
class CpuBudgetConfig:
    """Per-unit budget config matching the systemd resource-control vocabulary.

    See ADR-0004 §Decision and SDD-04 §Acceptance for the contract.
    """

    cpu_weight: int
    cpu_quota: str  # systemd-style "80%"
    memory_max: str  # "384M" or "1G"
    slice_name: str = "reachy-runtime.slice"
    nice: int = 0

    def __post_init__(self) -> None:
        if not (1 <= self.cpu_weight <= 10000):
            raise ValueError(f"cpu_weight must be in [1, 10000], got {self.cpu_weight}")
        if not (-20 <= self.nice <= 19):
            raise ValueError(f"nice must be in [-20, 19], got {self.nice}")
        # validate format eagerly so misconfig fails at construction
        parse_cpu_quota(self.cpu_quota)
        parse_memory_max(self.memory_max)


def parse_cpu_quota(raw: str) -> float:
    """Return CPU quota as a float fraction (1.0 == one full core).

    "80%" -> 0.80, "400%" -> 4.00. Raises ValueError on malformed input.
    """
    if not isinstance(raw, str):
        raise ValueError(f"CPUQuota must be a string, got {type(raw).__name__}")
    match = _CPU_QUOTA_RE.match(raw)
    if not match:
        raise ValueError(f"invalid CPUQuota {raw!r}; expected '<N>%'")
    return int(match.group(1)) / 100.0


def parse_memory_max(raw: str) -> int:
    """Return MemoryMax as a byte count. Accepts integer + M or G suffix."""
    if not isinstance(raw, str):
        raise ValueError(f"MemoryMax must be a string, got {type(raw).__name__}")
    match = _MEMORY_MAX_RE.match(raw)
    if not match:
        raise ValueError(f"invalid MemoryMax {raw!r}; expected '<N>M' or '<N>G'")
    value, suffix = int(match.group(1)), match.group(2)
    multiplier = 1024 * 1024 if suffix == "M" else 1024 * 1024 * 1024
    return value * multiplier


def validate_budget_total(configs: list[CpuBudgetConfig], cpu_cores: int = 4) -> list[str]:
    """Warn when summed CPUQuota exceeds 1.5x the available cores.

    CM4 has 4 cores by default. We allow some headroom (oversubscription is fine
    in cgroup v2 as long as workloads do not all peak together) but flag >= 1.5x.
    """
    warnings: list[str] = []
    total = sum(parse_cpu_quota(c.cpu_quota) for c in configs)
    threshold = cpu_cores * 1.5
    if total > threshold:
        warnings.append(
            f"cpu_quota_oversubscribed:{total:.2f}>threshold{threshold:.2f}_on_{cpu_cores}_cores"
        )
    return warnings


def validate_against_slice_cap(
    configs: list[CpuBudgetConfig],
    slice_cap_pct: float = 380.0,
) -> list[str]:
    """Return warnings if sum of per-service CPUQuota exceeds the slice CPUQuota cap.

    The shipped `deploy/systemd/reachy-runtime.slice` sets CPUQuota=380% on a
    4-core CM4 (one core reserved for kernel + daemon). When the sum of
    per-service CPUQuota strictly exceeds that slice cap, systemd cgroup v2 will
    silently throttle every service inside the slice — operators need a heads-up
    long before they see strange `cpu.stat` numbers.

    The contract:
    - empty `configs` -> `[]` (no warning)
    - sum `<` cap -> `[]`
    - sum `==` cap -> `[]` (the shipped layout sits exactly here on purpose)
    - sum `>` cap -> single warning string containing both percentages, prefixed
      with `slice_cap_exceeded:` so callers can grep for it.
    """
    if not configs:
        return []
    total_pct = sum(parse_cpu_quota(c.cpu_quota) for c in configs) * 100.0
    if total_pct > slice_cap_pct:
        return [
            f"slice_cap_exceeded:{total_pct:.0f}%>{slice_cap_pct:.0f}%_"
            f"(shipped reachy-runtime.slice CPUQuota={slice_cap_pct:.0f}%)"
        ]
    return []


def parse_systemd_unit(path: Path) -> configparser.RawConfigParser:
    """Permissive systemd unit parser. Returns RawConfigParser with case-preserved keys."""
    cp = configparser.RawConfigParser(strict=False, allow_no_value=True)
    cp.optionxform = str
    cp.read(path, encoding="utf-8")
    return cp
