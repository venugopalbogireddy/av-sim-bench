# sim-log-evaluator

A log-replay evaluation pipeline for simulated autonomous-driving data.
Ingests Parquet driving logs, computes 5 behavioural metrics, and emits a
dashboard image + JSON report — answering the core question every AV
simulation team asks: *"Did the simulator do the right thing?"*

---

## Problem

Simulation is the primary scaling lever for AV safety testing. Before a
simulator can be trusted, you need a repeatable, automated way to grade its
output against a known-good baseline. This project provides that harness:

- **Input**: columnar Parquet logs from any number of simulation runs
- **Processing**: 5 pure metric functions, each returning a structured `MetricResult(name, value, passed, details)`
- **Output**: `metrics.json` + a 4-panel dashboard PNG

---

## Architecture

```
src/evaluator/
  schema.py  → Parquet schema + map geometry constants (single source of truth)
  log_gen.py → synthetic scenario generator (PyArrow + NumPy)
  metrics.py → 5 pure metric functions (Pandas / SciPy)
  cli.py     → Click CLI wiring everything together

data/
  golden_YYYY_MM_DD.parquet      ← reference run (well-behaved agents)
  regression_YYYY_MM_DD.parquet  ← red-light runner + stop-sign roller
  noisy_YYYY_MM_DD.parquet       ← speed-jitter variant, otherwise compliant

outputs/
  metrics.json
  dashboard.png
```

```
┌─────────────┐          ┌──────────────┐   Parquet    ┌─────────────────┐   MetricResult[]   ┌─────────────┐
│  schema.py  │ ──────▶  │  log_gen.py  │ ──────────▶  │   metrics.py    │ ─────────────────▶ │   cli.py    │
│ (schema +   │          │ (3 scenarios)│              │  (5 pure fns)   │                    │  JSON + PNG │
│  geometry)  │ ──────▶  └──────────────┘              └─────────────────┘                    └─────────────┘
└─────────────┘
```

---

## Scenarios

| Scenario | Behaviour | Expected outcome |
|----------|-----------|-----------------|
| `golden` | Full stop at stop signs, obeys red lights, no collisions, nominal cruise speed | Passes all 5 metrics |
| `regression` | `agent_00` runs red light, `agent_01` rolls stop sign | Fails stop, red-light, collision metrics |
| `noisy` | Compliant at all signs and lights, cruise speed has higher variance (σ=1.2 vs σ=0.05) | Passes safety metrics, flagged by KS test |

---

## Metrics

| # | Name | Logic | Pass criterion |
|---|------|-------|----------------|
| 1 | `stop_sign_compliance` | Fraction of agents that held speed = 0 m/s for ≥ 10 consecutive frames (1 second) inside the stop zone | 1.0 — all agents must fully stop |
| 2 | `red_light_violation_rate` | Fraction of agents that moved (speed > 0 m/s) inside the intersection while light = RED; uses `in_intersection` schema flag | 0.0 — zero violations |
| 3 | `collision_proxy` | Count of log rows where `collision_flag == True` | 0 — no collision rows |
| 4 | `route_completion` | Fraction of agents whose maximum recorded y ≥ goal waypoint (50.0 m) | 1.0 — all agents reach the goal |
| 5 | `speed_ks_test` | Two-sample KS test p-value comparing cruise speeds vs baseline; cruise = rows outside stop zones and red-light states | p ≥ 0.05 — no distribution drift detected |

### Design decisions

**Stop compliance — sustained stop, not a speed dip.**
An agent that briefly touches 0 m/s then accelerates is not compliant. The metric requires 10 consecutive frames (1 second) at 0 m/s, matching the real-world legal standard for a complete stop.

**Red-light detection — schema flag, not position bounds.**
Detection uses the `in_intersection` boolean field written by the generator rather than a hardcoded y-coordinate range. This keeps the metric map-agnostic — if the scenario geometry changes, only `schema.py` needs updating, not the metric logic.

**KS test — cruise speeds only.**
Stop zones and red-light rows are excluded before running the KS test. Including them would confound deliberate slowdowns with genuine distribution drift in free-flow behaviour.

**Each metric is a pure function.**
Every metric takes a DataFrame and returns a `MetricResult`. No shared state, no side effects — easy to unit-test, parallelise, and version independently.

---

## Results

### 3-run comparison

| Metric | golden | noisy | regression |
|--------|--------|-------|------------|
| stop_sign_compliance | ✅ 1.00 | ✅ 1.00 | ❌ 0.75 |
| red_light_violation_rate | ✅ 0.00 | ✅ 0.00 | ❌ 0.50 |
| collision_proxy | ✅ 0 | ✅ 0 | ❌ 3 |
| route_completion | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| speed_ks_test (p-value) | ✅ 1.00 | ❌ 0.00 | ✅ 0.58 |

### Dashboard

![dashboard](outputs/dashboard.png)

- **regression** fails 3 safety-critical metrics — stop sign, red light, and collision
- **noisy** passes all safety metrics but is correctly flagged by the KS test; its cruise speed distribution is statistically distinguishable from the baseline even though no rules were broken
- **golden** passes all 5 metrics when evaluated against itself as baseline

---

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic logs
PYTHONPATH=src python -m evaluator.cli generate --out data/

# 3. Evaluate all logs against the golden baseline
PYTHONPATH=src python -m evaluator.cli run \
  --logs data/ \
  --baseline data/golden_$(date +%Y_%m_%d).parquet \
  --out outputs/

# 4. Run tests
pytest tests/ -v
```

---

## How to extend

| Goal | Where to change |
|------|----------------|
| Add a new metric | Add a pure function to `metrics.py`, register it in `compute_all()` |
| Change scenario behaviour | Edit the relevant scenario function in `log_gen.py` |
| Change map geometry | Update `STOP_ZONE_Y` / `INTERSECTION_Y` in `schema.py` — propagates to both generator and metrics |
| Use real logs | Replace the `generate` step with your own Parquet export; schema must match `schema.py` |
| CI | Add `PYTHONPATH=src pytest tests/` to your GitHub Actions workflow; no external services needed |

---

## Tech stack

Python 3.11+ · Pandas · PyArrow · NumPy · SciPy · Matplotlib · Click · Pytest
