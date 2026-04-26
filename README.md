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
Stop zones and red-light rows are excluded before running the KS test. Including them would confound deliberate slowdowns with genuine distribution drift in free-flow behaviour. The two-sample Kolmogorov-Smirnov statistic is distribution-free — it makes no assumption about the shape of the speed distribution, making it robust across scenarios where agents exhibit non-Gaussian or bimodal cruise profiles. A p-value ≥ 0.05 means we cannot reject the null hypothesis that both samples were drawn from the same underlying distribution.

**Each metric is a pure function.**
Every metric takes a DataFrame and returns a `MetricResult`. No shared state, no side effects — easy to unit-test, parallelise, and version independently. This composability is intentional: in a production eval pipeline, metric functions run in parallel across thousands of log shards; coupling them to shared state would make that impossible.

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
- **noisy** passes all safety metrics but is correctly flagged by the KS test; its cruise speed distribution is statistically distinguishable from the baseline even though no rules were broken. This illustrates the gap between rule-based and distributional evaluation: a run can be fully rules-compliant while silently regressing in behaviour — the kind of drift that accumulates undetected between simulator versions.
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

## Visualizations

### P1 — Agent trajectory GIF

Animates golden vs regression agents on the 1D track (stop zone + intersection).
Generates directly from `log_gen.py` — no Parquet files needed.

```bash
# Production output (small file, website-ready — 100 frames, 0.2 MB)
python scripts/gen_gif.py

# Higher fidelity for local inspection
python scripts/gen_gif.py --subsample 1 --fps 30 --dpi 120 --no-optimize

# Custom output path
python scripts/gen_gif.py --out /tmp/test.gif
```

| Flag | Default | Effect |
|------|---------|--------|
| `--subsample N` | `5` | Keep every Nth frame (5 → 100 frames from 500 ticks) |
| `--fps N` | `15` | Playback frame rate |
| `--dpi N` | `80` | Render resolution |
| `--no-optimize` | off | Skip PIL palette optimisation pass |
| `--out PATH` | `docs/assets/p1_agents.gif` | Output file location |

### P2 — Grid-world scenario visualization

Renders a 3-panel static figure (golden / noisy / regression) showing the 5×5 road
graph, all 4 agent paths, traffic-light state rings, and regression violation markers.

```bash
python scripts/gen_p2_viz.py
# Output: outputs/p2/p2_scenario_viz.png  (also copied to docs/assets/)
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

## Modifying a scenario and regenerating outputs

The full workflow when changing any P2 scenario config — edit, re-run pipeline, update filename reference, regenerate visualizations.

### Worked example: changing agent speed noise

**Step 1 — Edit the config**

Open `configs/p2_golden.yaml` (and `configs/p2_regression.yaml` if the same change applies there).
Change `speed_noise_std` for the relevant agent:

```yaml
# Before
- agent_id: agent_02
  speed_noise_std: 0.05   # tight, uniform noise

# After — more variable driver profile
- agent_id: agent_02
  speed_noise_std: 0.15
```

This was applied to give `agent_02` and `agent_03` distinct speed profiles (`0.15` and `0.30`
respectively), modelling realistic variation in a compliant fleet.

**Step 2 — Re-run the P2 pipeline**

```bash
PYTHONPATH=src python -m evaluator.cli pipeline run
```

This sims all three P2 scenarios and re-evaluates them. Regenerates:
- `data/p2/p2_*_YYYY_MM_DD.parquet` — new logs with today's date
- `outputs/p2/p2_metrics.json`
- `outputs/p2/p2_dashboard.png`

**Step 3 — Update the parquet filename in viz scripts**

The visualization scripts hardcode the parquet date. After a pipeline re-run,
update the date string in both scripts to match the new files:

```bash
# Confirm the new filename date
ls data/p2/

# Update in gen_p2_viz.py — find and replace the old date string (e.g. 2026_04_23 → 2026_04_25)
# Update in gen_p2_gif.py — same replacement
```

**Step 4 — Regenerate visualizations**

```bash
python scripts/gen_p2_viz.py   # → outputs/p2/p2_scenario_viz.png
python scripts/gen_p2_gif.py   # → docs/assets/p2_agents.gif

# Copy updated static viz to website assets
cp outputs/p2/p2_scenario_viz.png docs/assets/p2_scenario_viz.png
cp outputs/p2/p2_dashboard.png docs/assets/p2_dashboard.png
```

**Step 5 — Run tests**

```bash
pytest tests/ -v
```

All 86 tests should pass. Config changes affect simulation behaviour only —
metric logic is unchanged so a test failure indicates a broken sim re-run.

---

## Tech stack

Python 3.11+ · Pandas · PyArrow · NumPy · SciPy · Matplotlib · Click · Pytest

---

## References

- **CARLA Leaderboard Metrics** — industry-standard AV infraction taxonomy (collision, red-light, stop sign) that informed P1's metric suite. [leaderboard.carla.org](https://leaderboard.carla.org/)
- **Waymo Open Dataset** (Ettinger et al., 2021) — real-world driving log format, scenario taxonomy, and the production eval context this pipeline targets. [arxiv.org/abs/2104.10133](https://arxiv.org/abs/2104.10133)
- **Waymax: An Accelerated, Data-Driven Simulator** (Gulino et al., 2023) — Waymo's open-source simulator; shows how agent behaviour, counterfactuals, and planner evaluation are modelled at scale. [arxiv.org/abs/2310.08710](https://arxiv.org/abs/2310.08710)
- **Kolmogorov-Smirnov Test** — the distribution-free two-sample test used for cruise speed drift detection in metric 5. [en.wikipedia.org/wiki/Kolmogorov–Smirnov_test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
- **Benjamini-Hochberg FDR Procedure** — the multiple-testing correction used in P3's 5-test A/B framework; controls false discovery rate without the over-conservatism of Bonferroni correction. [en.wikipedia.org/wiki/False_discovery_rate](https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Hochberg_procedure)
