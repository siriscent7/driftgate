"""Seed DriftGate with a demo scenario: one stable feature, one drifting.
Usage: python seed.py <base_url>
Example: python seed.py http://localhost:8000
"""
import sys
import random
import time
import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def seed():
    with httpx.Client(base_url=BASE, timeout=10.0) as c:
        # Feature 1: "transaction_amount" — STABLE (live matches baseline)
        base1 = [random.gauss(100, 20) for _ in range(2000)]
        c.post("/baseline", json={"feature": "transaction_amount", "values": base1})
        for _ in range(300):
            c.post("/observe",
                   json={"feature": "transaction_amount", "value": random.gauss(100, 20)})

        # Feature 2: "user_age" — DRIFTING (live shifted from baseline)
        base2 = [random.gauss(35, 8) for _ in range(2000)]
        c.post("/baseline", json={"feature": "user_age", "values": base2})
        for _ in range(300):
            c.post("/observe",
                   json={"feature": "user_age", "value": random.gauss(48, 8)})  # shifted!

        t = time.perf_counter()
        report = c.get("/drift").json()
        elapsed = (time.perf_counter() - t) * 1000
        print(f"Drift computed in {elapsed:.0f} ms across {len(report)} features:")
        for feat, res in report.items():
            print(f"  {feat}: {res}")


if __name__ == "__main__":
    seed()