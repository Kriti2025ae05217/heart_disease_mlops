"""Send N requests to the API to populate logs/metrics for the monitoring demo.

    python scripts/load_test.py --url http://localhost:8000 -n 200
"""
import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

base = json.loads((Path(__file__).parent / "sample_request.json").read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("-n", type=int, default=100)
    a = p.parse_args()
    for _ in range(a.n):
        body = dict(base, age=random.randint(30, 75), chol=random.randint(150, 350),
                    thalach=random.randint(90, 195), exang=random.randint(0, 1))
        req = urllib.request.Request(a.url + "/predict", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req).read()
        time.sleep(0.05)
    print(f"sent {a.n} requests")


if __name__ == "__main__":
    main()
