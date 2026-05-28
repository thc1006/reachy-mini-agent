from reachy_intent_runtime.resource_policy import CpuBudgetPolicy


def test_cpu_over_budget_warns() -> None:
    warnings = CpuBudgetPolicy(max_cpu_percent=70).evaluate(cpu_percent=80, missed_heartbeats=0)
    assert warnings == ["cpu_over_budget:80.0>70.0"]


def test_heartbeat_deadline_warns() -> None:
    warnings = CpuBudgetPolicy(max_missed_heartbeats=2).evaluate(
        cpu_percent=10, missed_heartbeats=2
    )
    assert warnings == ["critical_loop_deadline_missed:2>=2"]
