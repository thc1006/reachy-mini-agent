from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CpuBudgetPolicy:
    """Lightweight policy object for Pi-side runtime budget decisions."""

    max_cpu_percent: float = 75.0
    critical_heartbeat_ms: int = 250
    max_missed_heartbeats: int = 2

    def evaluate(self, cpu_percent: float, missed_heartbeats: int) -> list[str]:
        warnings: list[str] = []
        if cpu_percent > self.max_cpu_percent:
            warnings.append(f"cpu_over_budget:{cpu_percent:.1f}>{self.max_cpu_percent:.1f}")
        if missed_heartbeats >= self.max_missed_heartbeats:
            warnings.append(
                f"critical_loop_deadline_missed:{missed_heartbeats}>={self.max_missed_heartbeats}"
            )
        return warnings
