"""Flask model-serving API.

Endpoints:
    GET  /health   liveness/readiness
    POST /predict  JSON features -> prediction + confidence
    GET  /metrics  Prometheus metrics
"""
import json
import logging
import os
import time
import uuid
from pathlib import Path

from flask import Flask, g, jsonify, request

from src.heart.predict import Predictor, ValidationError

from .metrics import Metrics

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
log = logging.getLogger("heart-api")


def create_app(model_dir=None) -> Flask:
    app = Flask(__name__)
    model_dir = Path(model_dir or os.getenv("MODEL_DIR", "models"))
    predictor = Predictor(model_dir)
    metrics = Metrics()

    @app.before_request
    def _start():
        g.t0 = time.perf_counter()
        g.request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])

    @app.after_request
    def _finish(resp):
        dt = time.perf_counter() - g.t0
        path = request.url_rule.rule if request.url_rule else "unmatched"
        metrics.observe_request(request.method, path, resp.status_code, dt)
        log.info(json.dumps({"event": "request", "id": g.request_id,
                             "method": request.method, "path": request.path,
                             "status": resp.status_code,
                             "latency_ms": round(dt * 1000, 2)}))
        resp.headers["X-Request-ID"] = g.request_id
        return resp

    @app.get("/health")
    def health():
        return jsonify(status="ok",
                       model=predictor.metadata.get("selected_model", "unknown"))

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        try:
            result = predictor.predict(payload)
        except ValidationError as exc:
            log.warning(json.dumps({"event": "validation_error", "id": g.request_id,
                                    "detail": str(exc)}))
            return jsonify(error=str(exc)), 400
        metrics.observe_prediction(result["prediction"])
        log.info(json.dumps({"event": "prediction", "id": g.request_id,
                             "prediction": result["prediction"],
                             "confidence": result["confidence"]}))
        return jsonify(result)

    @app.get("/metrics")
    def prom():
        return metrics.render(), 200, {"Content-Type": "text/plain; version=0.0.4"}

    return app


def app_factory():  # gunicorn "api.app:app_factory()"
    return create_app()
