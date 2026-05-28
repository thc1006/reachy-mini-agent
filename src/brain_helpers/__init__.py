"""brain_helpers — small in-process sidecars for the brain process.

Modules here are loaded by robot_brain.py main() AFTER brain_ready, and
spawn daemon threads that share PID with the brain (so they can use
os.sched_setscheduler() against sibling threads without CAP_SYS_NICE).

Current members:
  - cap_gupnp_thread  Wave4-P11 #85 Option (c) — pin libgupnp-igd's GMainLoop
                      worker thread to SCHED_IDLE on one allowed core, to
                      neutralise the Tailscale-userspace UPnP spin.
"""
