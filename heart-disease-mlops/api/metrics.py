"""Minimal, dependency-free Prometheus metrics (text exposition format 0.0.4)."""
import threading
from collections import defaultdict

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.requests = defaultdict(int)       # (method, path, status) -> n
        self.predictions = defaultdict(int)    # class -> n
        self.latency_bucket = defaultdict(int)  # (path, le) -> n
        self.latency_sum = defaultdict(float)
        self.latency_count = defaultdict(int)
        self.errors = 0

    def observe_request(self, method, path, status, seconds):
        with self._lock:
            self.requests[(method, path, str(status))] += 1
            if status >= 500:
                self.errors += 1
            for b in BUCKETS:
                if seconds <= b:
                    self.latency_bucket[(path, str(b))] += 1
            self.latency_bucket[(path, "+Inf")] += 1
            self.latency_sum[path] += seconds
            self.latency_count[path] += 1

    def observe_prediction(self, cls: int):
        with self._lock:
            self.predictions[str(cls)] += 1

    def render(self) -> str:
        with self._lock:
            out = ["# HELP http_requests_total Total HTTP requests",
                   "# TYPE http_requests_total counter"]
            for (m, p, s), n in sorted(self.requests.items()):
                out.append(f'http_requests_total{{method="{m}",path="{p}",status="{s}"}} {n}')
            out += ["# HELP http_request_duration_seconds Request latency",
                    "# TYPE http_request_duration_seconds histogram"]
            for (p, le), n in sorted(self.latency_bucket.items(),
                                     key=lambda kv: (kv[0][0], float(kv[0][1]))):
                out.append(f'http_request_duration_seconds_bucket{{path="{p}",le="{le}"}} {n}')
            for p, v in sorted(self.latency_sum.items()):
                out.append(f'http_request_duration_seconds_sum{{path="{p}"}} {v:.6f}')
                out.append(f'http_request_duration_seconds_count{{path="{p}"}} '
                           f'{self.latency_count[p]}')
            out += ["# HELP predictions_total Predictions by class",
                    "# TYPE predictions_total counter"]
            for c, n in sorted(self.predictions.items()):
                out.append(f'predictions_total{{class="{c}"}} {n}')
            out += ["# HELP http_errors_total 5xx responses",
                    "# TYPE http_errors_total counter",
                    f"http_errors_total {self.errors}"]
        return "\n".join(out) + "\n"
