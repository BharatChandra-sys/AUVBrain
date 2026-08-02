# AUVBrain — Hardening Roadmap

Honest scope. This is a strong solo-built autonomy stack (async control loop,
timeout-bounded I/O, safety override, LLM decision engine with rule fallback).
This roadmap is what actually moves it from "good portfolio project" to
"defensible, hardened system" — not a wishlist. Every item here is something
one person can realistically build and verify.

**Not included on purpose:** formal functional-safety certification (IEC 61508 /
ISO 13849), FMEA sign-off, multi-vendor redundant hardware, and anything that
requires a certified safety engineer or a real dive-test program. Those are
legitimate next steps for a *funded team*, not a solo checklist item — claiming
otherwise on this repo would be a liability, not a flex, given it drives real
thrusters.

---

## P0 — Safety-Critical Gaps (do these first, before anything else)

- [x] Add authentication to `/mode` and `/command` REST endpoints (API key or JWT) — `src/auvbrain/auth/dependencies.py` + `require_scope("write")` on all write routes
- [x] Add authentication/handshake to `/ws/control` WebSocket — WS handler now rejects with close code 4001 before accept if key is missing/invalid
- [x] Add rate limiting on command endpoints to prevent command flooding — per-IP token-bucket in `src/auvbrain/api/limiter.py`, wired to POST /mode and POST /command
- [ ] Add a hardware-level watchdog independent of the software stack (e.g. a timer relay that cuts thruster power if no heartbeat in N seconds) — software-only timeouts aren't sufficient for a device that can be in water unattended
- [x] Write an explicit test for the MANUAL-mode command-application path — `tests/test_agent_loop.py::test_manual_update_command_tracked_in_state` + manual-mode tick test
- [x] Add bounds/sanity validation on `ThrusterCommand` values *before* they reach `hw.thrusters.apply()` — Pydantic `ge`/`le` bounds on `ThrusterCommand` + `ExperimentCommand.params` validator in `models.py`
- [ ] Document and test the failure path when `hw.thrusters.apply()` itself throws mid-tick (not just timeout)

## P1 — Testing & CI (currently ~6 tests, no CI test run)

- [x] Add a GitHub Actions workflow that runs `pytest` on every push/PR — `.github/workflows/ci.yml` runs on every push/PR across Python 3.10/3.11/3.12 with ruff lint + pytest-cov
- [x] Add tests for `SafetyMonitor.enforce` covering every branch (depth, battery, pressure, obstacle-stop) — `tests/test_safety_monitor.py` covers all 8 branches including obstacle command-only override and None passthrough
- [x] Add tests for `FallbackDecisionEngine` (primary timeout → fallback, cooldown trigger, cooldown expiry, consecutive-failure counting) — `tests/test_fallback_engine.py`
- [x] Add tests for the agent loop's SAFE-mode and MANUAL-mode branches directly — `tests/test_agent_loop.py`
- [x] Add a test for LLM output parsing failure paths — `test_fallback_engine.py::test_parse_fail_treated_as_failure`
- [x] Add coverage reporting (`pytest-cov`) with a real threshold — `pytest-cov>=5.0.0` in dev deps, `fail_under=70` in `pyproject.toml`
- [x] Add a smoke test that boots the full stack in SIM mode and asserts telemetry is written — `test_agent_loop.py::test_autonomous_mode_writes_tick_events`

## P2 — Finish or Remove Half-Built Features

- [x] `memory/rag.py`: `ProtocolRetriever.retrieve()` always returns `[]` — now backed by the DB (`RagDocumentRepository.search_by_content`), with `ingest()` and `list_documents()` methods; pgvector ANN is the documented next step for embeddings
- [ ] Verify `raspi_gpio.py` against actual hardware and record the result (photo/video + telemetry log) — right now there's no evidence in-repo that this path has been run on a Pi
- [x] Confirm the two thruster layouts (`VECTORED_4_HORIZONTAL`, `H2_V2`) are both tested with real mixer unit tests — `tests/test_mixer_4motor.py` now covers both layouts across 8 test cases

## P3 — Observability (real, not aspirational)

- [x] Add structured logging with correlation IDs per tick — `logging_config.py` emits JSON logs; each tick sets `correlation_id` via `contextvars.ContextVar`, propagated to all log lines and telemetry events
- [x] Expose a `/metrics` endpoint — `GET /metrics` returns tick count, dropped telemetry, current mode, LLM fallback rate, decide/tick latency percentiles (p50/p95/p99) from the in-process `MetricsRegistry`
- [x] Add an alert path when `SafetyMonitor` forces SAFE — logs at ERROR level + optional webhook via `AUV_ALERT_WEBHOOK_URL` (fire-and-forget async POST)
- [x] Track and expose LLM decision latency percentiles — `METRICS.record_decide_latency()` called every autonomous tick; exposed in `/metrics` response

## P4 — API & Architecture Hardening

- [x] Add input validation limits on `VehicleCommand.note` and experiment `params` — `note` max 256 chars, `params` max 16 scalar keys enforced by Pydantic validators in `models.py`
- [x] Add a dead-man's-switch pattern to MANUAL mode — `AUV_MANUAL_DEADMAN_S` (default 10s); if no new command arrives within that window `STATE.check_manual_deadman()` returns True and the loop transitions to SAFE
- [x] Version the command schema explicitly — `schema_version` field on `VehicleCommand`; mismatches raise `ValidationError` at the API boundary
- [x] Add graceful shutdown handling (SIGTERM) — `agent/main.py` installs `signal.signal` handlers for SIGTERM/SIGINT; sets `stop_event`, forces SAFE, flushes telemetry, writes `mission_end` event

## P5 — Documentation Honesty Pass

- [ ] Update README to clearly separate "implemented and tested" vs "implemented but unverified on hardware" vs "stub/planned" — currently everything reads as equally complete
- [ ] Add a known-limitations section (no auth on control endpoints until P0 is done, RAG is a stub, Pi path unverified)
- [ ] Keep the latency benchmark claims (p95 ≈ 0.839ms etc.) — those are real and good, don't touch them

---

## Deliberately Deferred (real, but not solo-scope right now)

These are legitimate directions if this ever becomes a funded/team project —
listed here so they're not forgotten, but not pretending they're near-term:

- Formal FMEA / hazard analysis
- Redundant sensor voting (multiple depth/pressure sensors cross-checked)
- Independent hardware safety controller (separate MCU, not just a relay watchdog)
- In-water test program with logged dive data
- Multi-vehicle fleet coordination
- Regulatory/compliance review if used beyond hobbyist/competition context

---

**Last updated:** 2026-08-02
**Current honest state:** P0 auth/rate-limiting done, P1 full test suite + CI workflow done (SafetyMonitor all branches, FallbackDecisionEngine, agent loop SAFE/MANUAL, coverage infra, pytest runs on every push/PR), P2 RAG backed by DB (full-text; pgvector ANN is next), P3 structured JSON logs + correlation IDs + /metrics endpoint + safety alert webhook, P4 schema versioning + dead-man's switch + SIGTERM graceful shutdown all done. Pi hardware path still unverified on real hardware (P2).
**Next milestone:** Verify Pi GPIO path on real hardware.
