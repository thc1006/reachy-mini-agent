# Security Policy

## Supported versions

This project is pre-1.0. Only the latest commit on `master` is maintained. Fixes for security issues will always target `master` and, where feasible, a patch release.

## Reporting a vulnerability

**Please do not file a public issue for security vulnerabilities.** Instead, open a [private security advisory](https://github.com/thc1006/reachy-mini-agent/security/advisories/new) on GitHub. This delivers the report directly to the maintainer and allows us to coordinate a fix before disclosure.

If GitHub advisories are not available to you, contact the maintainer via the email listed on their [GitHub profile](https://github.com/thc1006).

### What to include

- A description of the vulnerability and its potential impact.
- Steps to reproduce (or a proof-of-concept).
- Any mitigations you've already identified.
- Whether you plan to disclose publicly, and on what timeline.

### What to expect

- **Acknowledgement within 7 days.**
- **Triage + proposed remediation plan within 30 days** for issues with exploitable impact on users running the default configuration.
- **Public advisory at fix-release time**, crediting the reporter unless anonymity is requested.

## Scope

In scope:

- Authentication / authorization bypass in any service shipped here (`robot_brain.py`, `kokoro_server.py`, `whisper_server.py`).
- Command injection, SSRF, or similar issues triggerable via model input or network traffic.
- Prompt injection that escapes the camera-view sandbox and takes meaningful action.

Out of scope:

- Vulnerabilities in upstream dependencies — please report those upstream (Pollen Robotics, Ollama, edge-tts, etc.). File an issue here only if you need help identifying the right upstream.
- Anything that requires local shell access to the brain host as a prerequisite.
- Denial-of-service against the robot by physically crowding or shouting — the robot is a friendly cube, not a security appliance.
