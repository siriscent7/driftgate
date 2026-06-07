# 📊 DriftGate

**A drift-detection control plane that catches model decay before your users do.**

ML models degrade silently as production data drifts from training data.
DriftGate monitors feature distributions in real time, quantifies drift with
industry-standard statistics (PSI + KS-test), exposes Prometheus metrics, and
alerts when drift becomes significant.

🔗 **Live dashboard:** https://driftgate.onrender.com/dashboard
📖 **Interactive API docs:** https://driftgate.onrender.com/docs

---

## The problem
A model trained on last year's data slowly becomes wrong as the real world
shifts ("data drift"). Teams usually find out from dropping business metrics or
angry users — not from monitoring. DriftGate makes drift a first-class,
observable signal.

## What it does
- **PSI (Population Stability Index)** — industry-standard drift magnitude
  (`<0.1` stable, `0.1–0.25` moderate, `>0.25` significant).
- **KS-test** — complementary statistical test for distribution change.
- **Inline monitoring** — apps POST live feature values; DriftGate keeps a
  thread-safe rolling window and computes drift on demand.
- **Prometheus metrics** — `/metrics` exposes PSI per feature for Grafana
  dashboards and alerting (the same pattern as Vertex AI Model Monitoring).
- **Alert webhook** — fires on significant drift; wrapped so alerting failures
  never break the request path.
- **Live HTML dashboard** — color-coded drift status, auto-refreshing.

## Demo
Seeded with one stable and one drifting feature:
Drift computed in 264 ms across 2 features:
transaction_amount: stable (psi 0.0346) -- live matches training
user_age: significant (psi 2.5207) -- distribution shifted

DriftGate distinguishes a stable feature from a genuinely drifting one.
*(Timing includes network round-trips to managed hosting; local compute is sub-10ms.)*

## API
| Endpoint | Purpose |
|---|---|
| `POST /baseline` | Register a training-data baseline for a feature |
| `POST /observe`  | Submit a live feature value |
| `GET /drift`     | Per-feature drift report (PSI, KS, status) |
| `GET /dashboard` | Live color-coded HTML dashboard |
| `GET /metrics`   | Prometheus-format metrics |
| `GET /health`    | Health check |
| `GET /docs`      | Interactive Swagger UI |

## Architecture
```
App / model inference
│ POST /observe {feature, value}
▼
DriftMonitor ── thread-safe rolling window per feature
│
▼ GET /drift
PSI + KS vs. baseline ── classify: stable / moderate / significant
│
├── /metrics → Prometheus → Grafana
├── /dashboard → live HTML view
└── significant → alert webhook
```

## Run locally
```bash
git clone https://github.com/siriscent7/driftgate.git
cd driftgate
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --port 8000

# in another terminal — seed the demo scenario:
python seed.py http://localhost:8000
# then open http://localhost:8000/dashboard
```

## Tests
```bash
pytest -v   # 9 tests: PSI/KS detection + API behavior
```

## Tech stack
Python · FastAPI · NumPy/SciPy · Prometheus client · Docker · Render

## Limitations
Numeric features only (categorical drift would need different binning).
In-memory state — baselines/observations reset on restart; re-run seed.py to repopulate the live demo after a cold start.

## Future Work
- Persistent baseline storage (Redis/Postgres) so state survives restarts.
- Categorical-feature drift support.
- Per-feature configurable thresholds + scheduled drift evaluation.
- Concept-drift detection (target/label drift, not just feature drift).