import random
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_baseline_and_no_drift():
    base = [random.gauss(0, 1) for _ in range(2000)]
    r = client.post("/baseline", json={"feature": "f1", "values": base})
    assert r.status_code == 200

    # feed similar values -> stable
    for _ in range(200):
        client.post("/observe", json={"feature": "f1", "value": random.gauss(0, 1)})

    report = client.get("/drift").json()
    assert report["f1"]["status"] in ("stable", "moderate")


def test_baseline_and_drift_detected():
    base = [random.gauss(0, 1) for _ in range(2000)]
    client.post("/baseline", json={"feature": "f2", "values": base})

    # feed drifted values -> significant
    for _ in range(200):
        client.post("/observe", json={"feature": "f2", "value": random.gauss(4, 1)})

    report = client.get("/drift").json()
    assert report["f2"]["status"] == "significant"


def test_insufficient_data():
    base = [random.gauss(0, 1) for _ in range(2000)]
    client.post("/baseline", json={"feature": "f3", "values": base})
    client.post("/observe", json={"feature": "f3", "value": 0.5})  # only 1 sample
    report = client.get("/drift").json()
    assert report["f3"]["status"] == "insufficient_data"