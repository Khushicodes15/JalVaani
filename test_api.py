"""
JalVaani AI API — test suite.
Run the API first:  cd jalvaani_api && uvicorn main:app --reload --port 8000
Then from the repo root:  python test_api.py
"""
import json
import sys

import requests

BASE = "http://localhost:8000"
DELHI = {"latitude": 28.6, "longitude": 77.2, "date": "2020-08-15"}
RAJASTHAN = {"latitude": 26.0, "longitude": 73.0, "date": "2019-05-10"}
KERALA = {"latitude": 9.5, "longitude": 76.5, "date": "2018-01-20"}

passed = 0
failed = 0


def run(label: str, method: str, url: str, expected_status: int = 200, **kwargs):
    global passed, failed
    try:
        resp = getattr(requests, method.lower())(url, timeout=30, **kwargs)
        status = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = resp.text[:300]

        ok = status == expected_status
        tag = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"\n[{tag}] {label}")
        print(f"  {method.upper()} {url} → {status} (expected {expected_status})")
        if isinstance(body, dict):
            s = json.dumps(body, indent=2)
            print(s[:600] + (" ..." if len(s) > 600 else ""))
        else:
            print(f"  {body}")
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] {label} — cannot connect to {BASE}. Is uvicorn running?")
        failed += 1
    except Exception as exc:
        print(f"\n[ERROR] {label} — {exc}")
        failed += 1


# ── Test cases ────────────────────────────────────────────────────────────────

print("=" * 65)
print("JalVaani AI API — Test Suite")
print(f"Target: {BASE}")
print("=" * 65)

run("GET /  (root info)",
    "GET", f"{BASE}/")

run("GET /health  (model status)",
    "GET", f"{BASE}/health")

run("POST /predict/depth  (Delhi, monsoon)",
    "POST", f"{BASE}/predict/depth", json=DELHI)

run("POST /predict/depth  (Rajasthan, pre-monsoon)",
    "POST", f"{BASE}/predict/depth", json=RAJASTHAN)

run("POST /predict/contamination  (Delhi)",
    "POST", f"{BASE}/predict/contamination", json=DELHI)

run("POST /predict/contamination  (Kerala — low-risk zone)",
    "POST", f"{BASE}/predict/contamination", json=KERALA)

run("GET /forecast/RAMGARH1  (known Rajasthan station)",
    "GET", f"{BASE}/forecast/RAMGARH1")

run("GET /forecast/Rampura  (known Punjab station)",
    "GET", f"{BASE}/forecast/Rampura")

run("GET /forecast/nonexistent_xyz  (expect 404)",
    "GET", f"{BASE}/forecast/nonexistent_xyz", expected_status=404)

run("POST /report/full  (Delhi — full integrated report)",
    "POST", f"{BASE}/report/full", json=DELHI)

run("GET /stations  (page 1, 5 per page)",
    "GET", f"{BASE}/stations?page=1&per_page=5")

run("GET /stations/search?state=Rajasthan",
    "GET", f"{BASE}/stations/search?state=Rajasthan")

run("GET /stations/search?state=Kerala&district=Alappuzha",
    "GET", f"{BASE}/stations/search?state=Kerala&district=Alappuzha")

run("GET /stats/national",
    "GET", f"{BASE}/stats/national")

run("POST /predict/depth  (invalid lat — expect 422)",
    "POST", f"{BASE}/predict/depth",
    json={"latitude": 50.0, "longitude": 77.2, "date": "2020-08-15"},
    expected_status=422)

run("POST /predict/depth  (invalid date — expect 422)",
    "POST", f"{BASE}/predict/depth",
    json={"latitude": 28.6, "longitude": 77.2, "date": "not-a-date"},
    expected_status=422)

# ── Summary ───────────────────────────────────────────────────────────────────

total = passed + failed
print("\n" + "=" * 65)
print(f"Results: {passed} passed / {failed} failed / {total} total")
if failed == 0:
    print("All tests passed.")
else:
    print("Some tests failed — check model files in jalvaani_api/saved_models/ and data/.")
print("=" * 65)

sys.exit(0 if failed == 0 else 1)
