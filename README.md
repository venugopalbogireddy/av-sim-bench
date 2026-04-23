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
- **Processing**: 5 metric functions, each returning a structured
  `MetricResult(name, value, passed, details)`
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
  golden_YYYY_MM_DD.parquet     ← reference (well-behaved)
  regression_YYYY_MM_DD.parquet ← red-light runner + stop-sign roller
  noisy_YYYY_MM_DD.parquet      ← speed-jitter variant

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

## Metrics

| # | Name | Logic | Pass criterion |
|---|------|-------|----------------|
| 1 | `stop_sign_compliance` | Fraction of agents that held speed == 0 m/s for ≥ 10 consecutive frames (~1 second) inside the stop zone | 1.0 — all agents must fully stop |
| 2 | `red_light_violation_rate` | Fraction of agents that moved (speed > 0 m/s) inside the intersection while light = RED | 0.0 — zero violations |
| 3 | `collision_proxy` | Count of rows where `collision_flag == True` | 0 — no collisions |
| 4 | `route_completion` | Fraction of agents whose max y ≥ goal waypoint | 1.0 — all agents finish |
| 5 | `speed_ks_test` | Two-sample KS p-value comparing cruise speeds vs baseline (excludes stop zones + red-light rows) | p ≥ 0.05 — no distribution drift |

**Key design decisions:**
- Stop compliance requires a *sustained* stop (dwell frames), not a momentary speed dip — matching the real-world legal standard
- Red-light detection uses the `in_intersection` schema flag, not hardcoded position bounds — keeping metrics map-agnostic
- KS test filters to cruise-only speeds to avoid confounding deliberate slowdowns with distribution drift

---

## Results

### 3-Run Comparison

| Metric | golden | noisy | regression |
|--------|--------|-------|------------|
| stop_sign_compliance | ✅ 1.00 | ✅ 1.00 | ❌ 0.75 |
| red_light_violation_rate | ✅ 0.00 | ✅ 0.00 | ❌ 0.50 |
| collision_proxy | ✅ 0 | ✅ 0 | ❌ 3 |
| route_completion | ✅ 1.00 | ✅ 1.00 | ✅ 1.00 |
| speed_ks_test (p-value) | ✅ 1.00 | ❌ 0.00 | ✅ 0.58 |

### Dashboard

![dashboard](outputs/dashboard.png)

Key observations:
- **regression** correctly fails 3 safety-critical metrics (stop, red-light, collision)
- **noisy** passes all violation metrics but is flagged by the KS test — its speed
  distribution is statistically distinguishable from the baseline
- **golden** passes all 5 metrics when used as its own baseline

---

## How to run

```bash
# 1. Create + activate venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate synthetic logs
python -m evaluator.cli generate --out data/

# 3. Evaluate
python -m evaluator.cli run \
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
| Add a new metric | Add a function to `metrics.py`, register it in `compute_all()` |
| Change scenario behaviour | Edit scenario functions in `log_gen.py` |
| Change map geometry | Update constants in `schema.py` — propagates to generator and metrics |
| Real logs | Replace `generate` step with your Parquet export; schema must match `schema.py` |
| DuckDB at scale | `import duckdb; duckdb.query("SELECT * FROM parquet_scan('data/*.parquet')")` |
| CI | Add `pytest tests/` to GitHub Actions; no external services needed |

---

## Tech stack

Python 3.11+ · Pandas · PyArrow · NumPy · SciPy · Matplotlib · Click · Pytest
