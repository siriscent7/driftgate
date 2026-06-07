import os
import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, HTMLResponse
from pydantic import BaseModel
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
import random

from driftgate.baseline import DriftMonitor

app = FastAPI(title="DriftGate", description="ML feature-drift monitoring service")
monitor = DriftMonitor()

# Prometheus metric: PSI per feature
psi_gauge = Gauge("driftgate_feature_psi", "PSI per feature", ["feature"])

ALERT_WEBHOOK = os.getenv("ALERT_WEBHOOK_URL", "")


# ---- request models ----
class BaselineRequest(BaseModel):
    feature: str
    values: list[float]


class ObserveRequest(BaseModel):
    feature: str
    value: float


# ---- endpoints ----
@app.get("/")
def root():
    return {
        "service": "DriftGate",
        "description": "ML feature-drift monitoring",
        "endpoints": ["/baseline", "/observe", "/drift", "/metrics", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/baseline")
def set_baseline(req: BaselineRequest):
    monitor.set_baseline(req.feature, req.values)
    return {"feature": req.feature, "baseline_size": len(req.values)}


@app.post("/observe")
def observe(req: ObserveRequest):
    monitor.observe(req.feature, req.value)
    return {"feature": req.feature, "ok": True}


@app.get("/drift")
async def drift():
    report = monitor.drift_report()
    # update Prometheus gauges + fire alerts on significant drift
    for feature, result in report.items():
        if "psi" in result:
            psi_gauge.labels(feature=feature).set(result["psi"])
            if result["status"] == "significant":
                await _fire_alert(feature, result)
    return report


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def _fire_alert(feature: str, result: dict):
    """POST an alert to a configured webhook when drift is significant."""
    if not ALERT_WEBHOOK:
        return
    payload = {
        "alert": "significant_drift",
        "feature": feature,
        "psi": result["psi"],
    }
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(ALERT_WEBHOOK, json=payload)
    except Exception:
        pass  # never let alerting break the request path


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    report = monitor.drift_report()

    rows = ""
    for feature, res in report.items():
        status = res.get("status", "unknown")
        psi = res.get("psi", "—")
        ks = res.get("ks_statistic", "—")
        samples = res.get("samples", 0)
        color = {
            "stable": "#16a34a",
            "moderate": "#d97706",
            "significant": "#dc2626",
            "insufficient_data": "#64748b",
        }.get(status, "#64748b")
        rows += f"""
        <tr>
          <td>{feature}</td>
          <td><span class="badge" style="background:{color}">{status}</span></td>
          <td>{psi}</td>
          <td>{ks}</td>
          <td>{samples}</td>
        </tr>"""

    if not rows:
        rows = '<tr><td colspan="5" style="text-align:center;color:#64748b">No features yet — POST a baseline to /baseline</td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="5">
<title>DriftGate Dashboard</title>
<style>
  body {{ font-family:-apple-system,system-ui,sans-serif; background:#0f172a;
         color:#e2e8f0; margin:0; padding:40px; }}
  .wrap {{ max-width:820px; margin:0 auto; }}
  h1 {{ font-size:1.8rem; margin-bottom:2px; }}
  .tag {{ color:#94a3b8; margin-bottom:28px; }}
  table {{ width:100%; border-collapse:collapse; background:#1e293b;
          border-radius:10px; overflow:hidden; }}
  th,td {{ padding:14px 18px; text-align:left; border-bottom:1px solid #334155; }}
  th {{ background:#162133; color:#94a3b8; font-size:0.8rem; text-transform:uppercase; }}
  .badge {{ color:#fff; padding:3px 12px; border-radius:999px; font-size:0.8rem; }}
  .links {{ margin-top:24px; color:#94a3b8; font-size:0.9rem; }}
  a {{ color:#7dd3fc; }}
  .note {{ color:#64748b; font-size:0.8rem; margin-top:8px; }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>📊 DriftGate Dashboard</h1>
    <p class="tag">Live ML feature-drift monitoring · auto-refreshes every 5s</p>
    <table>
      <thead>
        <tr><th>Feature</th><th>Status</th><th>PSI</th><th>KS stat</th><th>Samples</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    <p class="links">
      <a href="/metrics">Prometheus metrics</a> ·
      <a href="/drift">Raw JSON</a> ·
      <a href="/docs">API docs</a>
    </p>
    <p class="note">PSI: &lt;0.1 stable · 0.1–0.25 moderate · &gt;0.25 significant</p>
  </div>
</body>
</html>"""
    return html

@app.on_event("startup")
def seed_demo_data():
    """Populate demo data on startup so the dashboard is never empty.
    Controlled by an env var so it can be disabled in real use."""
    if os.getenv("SEED_DEMO", "true").lower() != "true":
        return

    # stable feature: live matches baseline
    base1 = [random.gauss(100, 20) for _ in range(2000)]
    monitor.set_baseline("transaction_amount", base1)
    for _ in range(300):
        monitor.observe("transaction_amount", random.gauss(100, 20))

    # drifting feature: live shifted from baseline
    base2 = [random.gauss(35, 8) for _ in range(2000)]
    monitor.set_baseline("user_age", base2)
    for _ in range(300):
        monitor.observe("user_age", random.gauss(48, 8))  # shifted!