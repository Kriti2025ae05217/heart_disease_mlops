# Lab runbook (VS Code) - do these in order

Everything is already built and unit-tested. In the lab we only **run it, capture proof, and record the video**.
All commands run in the **VS Code integrated terminal** (`` Ctrl+` ``; new terminal = `+`; split = `Ctrl+Shift+5`).

## 0. VS Code setup (5 min)
**Machine needs:** Python 3.11+, Git, Docker Desktop (running), kubectl + Minikube (or Docker Desktop Kubernetes enabled).
Check in a terminal: `python --version`, `git --version`, `docker version`, `kubectl version --client`, `minikube version`.

1. Unzip the project, then **File > Open Folder** -> `heart-disease-mlops` (open the folder itself, not its parent).
2. Install extensions (Ctrl+Shift+X): **Python**, **Jupyter**, **Docker** (Microsoft), **Kubernetes** (Microsoft, optional - handy for screenshots), **GitHub Actions** (optional).
3. Terminal shell: Windows -> use **PowerShell** or **Git Bash** (dropdown next to `+`). Commands below show both where they differ.
4. Python interpreter: after step 1 of section 1, press `Ctrl+Shift+P` -> **Python: Select Interpreter** -> pick `.venv`.

### Terminal gotchas
| Issue | Fix |
|---|---|
| PowerShell: `curl` is an alias for Invoke-WebRequest | use **`curl.exe`** (or Git Bash) |
| PowerShell: `-H "Content-Type: application/json" -d @file` | works with `curl.exe`; use `@scripts/sample_request.json` |
| PowerShell blocks venv activation | `Set-ExecutionPolicy -Scope Process Bypass` then activate again |
| `&` / `eval $(...)` don't work in PowerShell | use separate terminals; see Minikube note in section 4 |

## 1. Local run (tasks 1-4)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1         # PowerShell     |  Git Bash: source .venv/Scripts/activate  |  Linux/Mac: source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/download_data.py
python -m src.heart.eda
python -m src.heart.train          # re-train HERE so the model matches the lab's scikit-learn version
flake8 src api tests scripts
pytest -v
mlflow ui                          # leave running in its own terminal; open http://127.0.0.1:5000
```
- **Notebooks:** open `notebooks/01_eda.ipynb` -> top-right **Select Kernel** -> `.venv` -> **Run All**. Do the same for `02_` and `03_`. Save them (`Ctrl+S`) so outputs are stored, then commit them.
- **Tests panel:** Testing icon (flask) in the sidebar -> configure **pytest** -> run all (good for a screenshot/video).
- Screenshots: `mlflow_runs.png` (experiment table), `mlflow_run_detail.png` (one run: params/metrics/artifacts).

## 2. GitHub + CI/CD (task 5)
Terminal (or Source Control panel: **Initialize Repository** -> message -> **Commit** -> **Publish Branch**):
```bash
git init && git add . && git commit -m "MLOps assignment"
git branch -M main && git remote add origin <your-repo-url> && git push -u origin main
```
Create the empty repo on github.com first. If push asks for login, sign in via the browser prompt or use a Personal Access Token as the password.
- Actions tab on GitHub -> green run -> `github_actions_success.png` (show all 3 jobs + artifacts).
- Fail-fast proof: put `assert 1 == 2` in a test in `tests/test_data.py`, commit and push, screenshot the red run as `github_actions_failure.png`, then **revert and push again** so the final run is green.

## 3. Docker (task 6)
```bash
docker build -t heart-api:latest .
docker run -d --name heart -p 8000:8000 heart-api:latest
curl.exe localhost:8000/health
curl.exe -X POST localhost:8000/predict -H "Content-Type: application/json" -d @scripts/sample_request.json
docker rm -f heart
```
(Git Bash/Linux: `curl` instead of `curl.exe`.) The Docker extension's sidebar also shows images/containers for the screenshot. Capture build + run + curl together: `docker_run.png`.

## 4. Kubernetes (task 7)
```bash
minikube start --driver=docker
```
Build the image **inside Minikube's Docker**:
- Git Bash/Linux: `eval $(minikube docker-env) && docker build -t heart-api:latest .`
- PowerShell: `& minikube -p minikube docker-env --shell powershell | Invoke-Expression` then `docker build -t heart-api:latest .`
- Docker Desktop Kubernetes: skip this, the normal local image is already visible.
```bash
kubectl apply -f k8s/deployment.yaml -f k8s/hpa.yaml
kubectl rollout status deploy/heart-api
kubectl get pods,svc,hpa                 # -> kubectl_get_all.png
```
Expose the LoadBalancer: open a **second terminal**, run `minikube tunnel` and leave it (Docker Desktop: not needed). Then:
```bash
curl.exe http://<EXTERNAL-IP>/health
curl.exe -X POST http://<EXTERNAL-IP>/predict -H "Content-Type: application/json" -d @scripts/sample_request.json   # -> predict_curl.png
```
If EXTERNAL-IP stays `<pending>`: run `minikube service heart-api --url` and use that URL (and mention it in the report).

## 5. Monitoring (task 8)
Use **one terminal per long-running command** (split terminal):
```bash
kubectl apply -f k8s/monitoring.yaml
kubectl port-forward svc/prometheus 9090:9090     # terminal A
kubectl port-forward svc/grafana 3000:3000        # terminal B  (login admin / admin)
python scripts/load_test.py --url http://<EXTERNAL-IP> -n 200     # terminal C
kubectl logs -l app=heart-api --tail=20           # -> api_logs.png
```
Grafana at http://localhost:3000: Connections -> add Prometheus data source (`http://prometheus:9090`) -> Dashboards -> Import -> upload `monitoring/grafana-dashboard.json` -> screenshot as `grafana_dashboard.png`.
Fallback if Kubernetes monitoring misbehaves: `docker compose up --build` (API :8000, Prometheus :9090, Grafana :3000).

## 6. Finish the report (task 9)
1. Overwrite the 9 red-bordered files in `screenshots/` with your real screenshots (same filenames, PNG). VS Code can preview them: click the file.
2. Edit `docs/REPORT.md` (preview with `Ctrl+Shift+V`): fill the `<...>` fields - name, BITS ID, repo URL, video link, cluster used, API access. Update numbers if you changed models or re-trained (`reports/model_comparison.csv`).
3. Rebuild: `cd docs && pandoc REPORT.md -o REPORT.docx --resource-path=.:..`
   No pandoc in the lab? Open `docs/REPORT.docx` in Word and use **Change Picture** on each placeholder, then **Save As PDF**.
4. Put the repo URL in `README.md`, commit and push (Source Control panel).

## 7. Video (~3-4 min) - record with Win+G (Xbox Game Bar), OBS, or Zoom/Teams screen share
1. VS Code Explorer tour: folders, README (30s). 2. Run `pytest` + show MLflow UI (45s). 3. Push a commit, show Actions run (45s).
4. `docker build/run` + curl (30s). 5. `kubectl get pods,svc`, curl `/predict` via the service (45s).
6. Load test, then Grafana + logs (45s).

## 8. Personalise before submitting (academic-integrity note)
Change something genuinely yours: e.g. add a third model (GradientBoosting) in `src/heart/train.py::candidate_models`, tweak grids, change log fields in `api/app.py`, restyle Grafana panels. Re-run tests, re-train, update the report numbers. Be ready to explain every file.

## Final checklist (marks)
- [ ] EDA figures, cleaned CSV, download script (5)  - [ ] 2+ models, CV metrics (8)  - [ ] MLflow runs (5)
- [ ] model.joblib + requirements + pipeline (7)  - [ ] tests + green Actions + artifacts (8)  - [ ] Docker proof (5)
- [ ] K8s manifests + screenshots (7)  - [ ] logs + Grafana (3)  - [ ] ~10-page report + repo link (2)  - [ ] video
