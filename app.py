import os
import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

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