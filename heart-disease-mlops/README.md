# Heart Disease Risk Prediction - End-to-End MLOps

Predicts presence of heart disease from the UCI Heart Disease (Cleveland) data and serves it
as a containerised, monitored REST API deployed on Kubernetes.

**Repository:** `<GitHub URL>`  |  **Report:** `docs/REPORT.md` / `docs/REPORT.docx`

## Project layout
```
scripts/download_data.py   download raw UCI file + write cleaned CSV
src/heart/                 data.py, features.py, train.py, evaluate.py, tracking.py, predict.py, eda.py
api/                       Flask app (/predict /health /metrics) + Prometheus metrics
tests/                     unittest/pytest suite (data, features, training, API)
Dockerfile, docker-compose.yml
k8s/                       deployment+service, ingress, hpa, monitoring (Prometheus/Grafana)
monitoring/                Grafana dashboard JSON, local Prometheus config
.github/workflows/ci.yml   lint -> test -> train -> docker build + smoke test
```

## 1. Setup (clean machine)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## 2. Get the data
```bash
python scripts/download_data.py
```
Downloads `processed.cleveland.data` from
<https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data>
(dataset page: <https://archive.ics.uci.edu/dataset/45/heart+disease>) and writes
`data/processed/heart_clean.csv`. If the script cannot reach UCI, download the file manually
into `data/raw/` and re-run the script.

## 3. EDA, training, tracking
```bash
python -m src.heart.eda          # figures -> reports/figures
python -m src.heart.train        # logs to ./mlruns, saves models/model.joblib
mlflow ui                        # http://127.0.0.1:5000
```

## 4. Tests and lint
```bash
flake8 src api tests scripts
pytest -v
```

## 5. Docker
```bash
docker build -t heart-api:latest .
docker run -p 8000:8000 heart-api:latest
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @scripts/sample_request.json
```
Example response:
```json
{"confidence": 0.5762, "label": "no heart disease likely", "prediction": 0, "probability_disease": 0.4238}
```

## 6. Kubernetes (Minikube or Docker Desktop)
```bash
# Minikube: build into the cluster's Docker daemon
eval $(minikube docker-env) && docker build -t heart-api:latest .
kubectl apply -f k8s/deployment.yaml -f k8s/hpa.yaml
minikube tunnel                       # in a second terminal (Docker Desktop: not needed)
kubectl get svc heart-api             # EXTERNAL-IP -> curl http://<ip>/health
# optional ingress:  minikube addons enable ingress && kubectl apply -f k8s/ingress.yaml
```
For GKE/EKS/AKS: push the image to your registry and change `image:` in `k8s/deployment.yaml`.

## 7. Monitoring
```bash
kubectl apply -f k8s/monitoring.yaml
kubectl port-forward svc/prometheus 9090:9090 &
kubectl port-forward svc/grafana 3000:3000          # login admin/admin
python scripts/load_test.py --url http://<service-ip> -n 200
```
In Grafana: Dashboards -> Import -> upload `monitoring/grafana-dashboard.json`.
Logs: `kubectl logs -l app=heart-api` (one JSON line per request / prediction).
Docker-only alternative: `docker compose up --build`.

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness/readiness |
| POST | `/predict` | 13 features as JSON -> `prediction`, `confidence` |
| GET | `/metrics` | Prometheus metrics |

Feature codes follow the UCI documentation (`cp` 1-4, `slope` 1-3, `ca` 0-3, `thal` 3/6/7).
