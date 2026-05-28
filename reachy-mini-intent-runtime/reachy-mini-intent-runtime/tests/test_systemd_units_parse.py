"""SDD-04 acceptance: every deploy/systemd/* unit declares the required QoS keys.

We parse systemd unit files with a permissive INI parser (RawConfigParser with
strict=False) and assert ADR-0004 contract:

- every .service declares Slice=reachy-runtime.slice
- every .service has CPUWeight=, CPUQuota=, MemoryMax=, Nice=, User= in [Service]
- every .service has NoNewPrivileges=yes and StartLimitBurst= in [Unit]
- the .slice file declares CPUAccounting / MemoryAccounting / IOAccounting
- realtime fields (CPUSchedulingPolicy=fifo/rr) are NOT actively set; if present
  they must be commented out
- no .service references the demo fixture-replay module in ExecStart (HIGH-B1)
- no .service uses the misleading `PartOf=<slice>` pattern (HIGH-B2)
- the slice does not declare `Before=slices.target` (HIGH-B2)
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest

DEPLOY_DIR = Path(__file__).resolve().parents[1] / "deploy" / "systemd"

SERVICE_UNITS = [
    "reachy-orchestrator.service",
    "reachy-audio-listener.service",
    "reachy-motion-worker.service",
    "reachy-camera-sampler.service",
    "reachy-llm-vlm-client.service",
]

SLICE_UNIT = "reachy-runtime.slice"


def _load(unit: str) -> configparser.RawConfigParser:
    cp = configparser.RawConfigParser(strict=False, allow_no_value=True)
    cp.optionxform = str  # preserve case
    cp.read(DEPLOY_DIR / unit, encoding="utf-8")
    return cp


def _active_lines(unit: str) -> list[str]:
    """Return non-comment, non-blank lines of a unit file (preserves section headers)."""
    raw = (DEPLOY_DIR / unit).read_text(encoding="utf-8")
    return [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_deploy_systemd_directory_exists() -> None:
    assert DEPLOY_DIR.is_dir(), f"missing deploy dir: {DEPLOY_DIR}"


def test_slice_unit_declares_accounting() -> None:
    cp = _load(SLICE_UNIT)
    assert cp.has_section("Slice"), "[Slice] section missing"
    assert cp.get("Slice", "CPUAccounting").lower() == "yes"
    assert cp.get("Slice", "MemoryAccounting").lower() == "yes"
    assert cp.get("Slice", "IOAccounting").lower() == "yes"


@pytest.mark.parametrize("unit", SERVICE_UNITS)
def test_service_unit_has_required_qos_keys(unit: str) -> None:
    cp = _load(unit)
    assert cp.has_section("Service"), f"{unit}: [Service] section missing"
    assert cp.has_option("Service", "Slice"), f"{unit}: Slice= missing"
    assert cp.get("Service", "Slice") == "reachy-runtime.slice", (
        f"{unit}: Slice must be reachy-runtime.slice"
    )
    assert cp.has_option("Service", "CPUWeight"), f"{unit}: CPUWeight= missing"
    assert cp.has_option("Service", "CPUQuota"), f"{unit}: CPUQuota= missing"
    # CPUWeight must be a valid integer in systemd range 1-10000
    weight = int(cp.get("Service", "CPUWeight"))
    assert 1 <= weight <= 10000, f"{unit}: CPUWeight {weight} out of range"
    # CPUQuota must end with %
    quota = cp.get("Service", "CPUQuota").strip()
    assert quota.endswith("%"), f"{unit}: CPUQuota {quota!r} must end with %"


@pytest.mark.parametrize("unit", SERVICE_UNITS)
def test_service_unit_has_memory_max(unit: str) -> None:
    """MED-B1: every service must declare MemoryMax= (M or G suffix)."""
    cp = _load(unit)
    assert cp.has_option("Service", "MemoryMax"), f"{unit}: MemoryMax= missing"
    mem = cp.get("Service", "MemoryMax").strip()
    assert mem and (mem.endswith("M") or mem.endswith("G")), (
        f"{unit}: MemoryMax {mem!r} must end with M or G"
    )


@pytest.mark.parametrize("unit", SERVICE_UNITS)
def test_service_unit_has_nice_in_valid_range(unit: str) -> None:
    """MED-B1: every service must declare Nice= within POSIX [-20, 19]."""
    cp = _load(unit)
    assert cp.has_option("Service", "Nice"), f"{unit}: Nice= missing"
    nice = int(cp.get("Service", "Nice"))
    assert -20 <= nice <= 19, f"{unit}: Nice {nice} out of POSIX range"


@pytest.mark.parametrize("unit", SERVICE_UNITS)
def test_service_unit_runs_as_reachy_user(unit: str) -> None:
    """MED-B1 + HIGH-B3: every service must run as User=reachy (not root)."""
    cp = _load(unit)
    assert cp.has_option("Service", "User"), f"{unit}: User= missing"
    assert cp.get("Service", "User") == "reachy", (
        f"{unit}: User must be 'reachy', got {cp.get('Service', 'User')!r}"
    )


@pytest.mark.parametrize("unit", SERVICE_UNITS)
def test_service_unit_has_no_new_privileges(unit: str) -> None:
    """MED-B1 + HIGH-B3: NoNewPrivileges=yes blocks setuid escalation in children."""
    cp = _load(unit)
    assert cp.has_option("Service", "NoNewPrivileges"), f"{unit}: NoNewPrivileges= missing"
    assert cp.get("Service", "NoNewPrivileges").lower() == "yes", (
        f"{unit}: NoNewPrivileges must be 'yes'"
    )


@pytest.mark.parametrize("unit", SERVICE_UNITS)
def test_service_unit_has_start_limit_in_unit_section(unit: str) -> None:
    """MED-B1 + HIGH-B3: StartLimitBurst= belongs in [Unit] per systemd.unit(5)."""
    cp = _load(unit)
    assert cp.has_section("Unit"), f"{unit}: [Unit] section missing"
    assert cp.has_option("Unit", "StartLimitBurst"), (
        f"{unit}: StartLimitBurst= missing in [Unit] section"
    )
    burst = int(cp.get("Unit", "StartLimitBurst"))
    assert burst > 0, f"{unit}: StartLimitBurst must be > 0"


@pytest.mark.parametrize("unit", SERVICE_UNITS)
def test_service_unit_has_working_directory(unit: str) -> None:
    cp = _load(unit)
    assert cp.has_option("Service", "WorkingDirectory"), f"{unit}: WorkingDirectory= missing"
    wd = cp.get("Service", "WorkingDirectory")
    assert wd == "/opt/reachy-mini-intent-runtime", f"{unit}: unexpected WorkingDirectory {wd!r}"


@pytest.mark.parametrize("unit", SERVICE_UNITS)
def test_service_unit_realtime_scheduling_not_active(unit: str) -> None:
    """Realtime SCHED_FIFO/RR must be commented out per ADR-0004 opt-in policy."""
    raw = (DEPLOY_DIR / unit).read_text(encoding="utf-8")
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # bare key=value lines only
        if "CPUSchedulingPolicy" in stripped and "=" in stripped:
            key, _, value = stripped.partition("=")
            value = value.strip().lower()
            assert value not in {"fifo", "rr"}, (
                f"{unit}: realtime scheduler must not be active by default"
            )


@pytest.mark.parametrize("unit", SERVICE_UNITS)
def test_service_unit_has_no_obvious_typos(unit: str) -> None:
    """Catch common typos like CPUWieght / CPUQouta / WeightCPU."""
    raw = (DEPLOY_DIR / unit).read_text(encoding="utf-8")
    forbidden = ["CPUWieght", "CPUQouta", "WeightCPU", "QoutaCPU", "MemroyMax"]
    for tok in forbidden:
        assert tok not in raw, f"{unit}: typo {tok!r} found"


def test_audio_listener_has_highest_priority_weight() -> None:
    """Audio listener must have the highest CPUWeight per ADR-0004 tier 1."""
    weights = {unit: int(_load(unit).get("Service", "CPUWeight")) for unit in SERVICE_UNITS}
    assert weights["reachy-audio-listener.service"] >= max(weights.values()), (
        f"audio listener must have top weight; got {weights}"
    )


def test_llm_vlm_client_has_lowest_priority_weight() -> None:
    """LLM/VLM client must be tier 3 (lowest)."""
    weights = {unit: int(_load(unit).get("Service", "CPUWeight")) for unit in SERVICE_UNITS}
    assert weights["reachy-llm-vlm-client.service"] <= min(weights.values()), (
        f"llm-vlm-client must have lowest weight; got {weights}"
    )


# ---------------------------------------------------------------------------
# HIGH-B1 regression guard: ExecStart must NOT replay the demo fixture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit", SERVICE_UNITS)
def test_service_execstart_does_not_invoke_demo_replay(unit: str) -> None:
    """HIGH-B1: demo.py exits 0 after replaying a fixture; systemd would mark
    the service `active (exited)` and never restart it, misleading operators."""
    cp = _load(unit)
    exec_start = cp.get("Service", "ExecStart")
    assert "reachy_intent_runtime.demo" not in exec_start, (
        f"{unit}: ExecStart must not invoke reachy_intent_runtime.demo "
        f"(one-shot fixture replay); got {exec_start!r}"
    )
    assert "demo.py" not in exec_start, (
        f"{unit}: ExecStart must not invoke demo.py; got {exec_start!r}"
    )


def test_llm_vlm_client_uses_dedicated_module() -> None:
    """HIGH-B1: the placeholder must point at the dedicated heartbeat stub."""
    cp = _load("reachy-llm-vlm-client.service")
    exec_start = cp.get("Service", "ExecStart")
    assert "reachy_intent_runtime.llm_vlm_client" in exec_start, (
        f"reachy-llm-vlm-client.service: ExecStart must invoke "
        f"reachy_intent_runtime.llm_vlm_client; got {exec_start!r}"
    )


# ---------------------------------------------------------------------------
# HIGH-B2 regression guard: no PartOf=*.slice, no Before=slices.target
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit", SERVICE_UNITS)
def test_service_unit_does_not_use_partof_slice(unit: str) -> None:
    """HIGH-B2: PartOf= is service<->service stop-propagation, not slice membership.
    Slice membership belongs in [Service] Slice=."""
    for line in _active_lines(unit):
        if line.startswith("PartOf="):
            value = line.split("=", 1)[1].strip()
            assert not value.endswith(".slice"), (
                f"{unit}: PartOf={value} is wrong semantics; "
                f"use Slice=reachy-runtime.slice in [Service] instead"
            )


def test_slice_unit_does_not_order_before_slices_target() -> None:
    """HIGH-B2: slices are activated on demand; ordering against slices.target
    is incorrect and confuses operators reading `systemctl list-dependencies`."""
    for line in _active_lines(SLICE_UNIT):
        assert not (line.startswith("Before=") and "slices.target" in line), (
            f"{SLICE_UNIT}: must not declare Before=slices.target; got {line!r}"
        )
