# Deployment & Operations Guide

> Complete guide for deploying, operating, and troubleshooting the ML Production Pipeline.

---

## 1. Prerequisites

Install the following before starting:

| Tool | Version | Install command (Windows) | Verify |
|---|---|---|---|
| Docker Desktop | Latest | [Download](https://www.docker.com/products/docker-desktop/) | `docker --version` |
| Java JDK | **11** | `winget install EclipseAdoptium.Temurin.11.JDK` | `java -version` |
| SBT | Latest | `winget install sbt.sbt` | `sbt --version` |
| Python | 3.10+ | [Download](https://www.python.org/downloads/) | `python --version` |
| Git | Latest | `winget install Git.Git` | `git --version` |

> ⚠️ **Use Java 11, not 17 or later.** Spark 3.4 has compatibility issues with newer Java versions.

---

## 2. First-Time Setup

### Clone the repository
```bash
git clone <your-repo-url>
cd project9-mlflow
```

### Create environment file
```bash
cp .env.example .env
# Edit .env to change passwords if desired (optional for local dev)
```

### Download the dataset
1. Go to [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Place it in `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### Install Python dependencies (for scripts)
```bash
pip install mlflow boto3 httpx locust
```

---

## 3. Start All Services

```bash
docker-compose up -d
```

Wait ~60 seconds for all services to initialize, then verify:

```bash
docker-compose ps
```

All containers should show `Up` or `Up (healthy)`.

### Service URLs

| Service | URL | Credentials |
|---|---|---|
| MLflow UI | http://localhost:5000 | None |
| Spark Master | http://localhost:8080 | None |
| FastAPI Docs | http://localhost:8000/docs | None |
| MinIO Console | http://localhost:9001 | `minioadmin` / `minioadmin_secret_2024` |
| Prometheus | http://localhost:9090 | None |
| Grafana | http://localhost:3000 | `admin` / `admin` |

---

## 4. Train the Model

### Option A: Run locally with SBT (development)
```bash
cd training
sbt run
```

This runs Spark in local mode and logs to the MLflow server on Docker.

### Option B: Build JAR and submit to Spark cluster
```bash
cd training
sbt assembly

# Submit to the Spark cluster
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --class Main \
  /path/to/churn-training.jar
```

Training takes 5–15 minutes depending on your hardware. You'll see progress printed to the console.

---

## 5. Check MLflow UI

1. Open http://localhost:5000
2. You should see 4 experiments:
   - `churn-exploration` — data exploration summary
   - `churn-logistic-regression` — LR training run
   - `churn-random-forest` — RF training run
   - `churn-gradient-boosted-trees` — GBT training run
3. Click any experiment to see logged parameters, metrics, and artifacts
4. Compare runs: select multiple runs → click "Compare"

---

## 6. Promote the Best Model

### Via the training pipeline (automatic)
The training script automatically registers the best model and promotes it to Production if AUC > 0.80.

### Via MLflow UI (manual)
1. Go to http://localhost:5000
2. Click "Models" in the top nav
3. Click `churn-predictor`
4. Select the version to promote
5. Click "Stage" → "Transition to Production"

### Via script
```bash
# Promote the latest Staging model
python scripts/promote_model.py

# Promote a specific version
python scripts/promote_model.py --version 2

# Force promote (skip validation)
python scripts/promote_model.py --version 2 --force
```

---

## 7. Verify the API

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35, "TotalCharges": 1397.475
  }'

# Swagger docs (interactive API explorer)
# Open http://localhost:8000/docs in your browser
```

---

## 8. Update the Model (Zero-Downtime)

To update the model without stopping the API:

1. **Retrain** with new data or hyperparameters:
   ```bash
   cd training && sbt run
   ```

2. **Compare** the new version against the current one:
   ```bash
   python scripts/compare_models.py --v1 1 --v2 2
   ```

3. **Promote** the new version if it's better:
   ```bash
   python scripts/promote_model.py --version 2
   ```

4. **Restart the API** to load the new model:
   ```bash
   docker-compose restart api
   ```

The API will load the new Production model on startup. During the restart (~5 seconds), requests will return 503.

---

## 9. Roll Back to a Previous Version

If the new model performs poorly in production:

1. **Demote** the current Production version:
   ```bash
   # In Python
   from mlflow.tracking import MlflowClient
   client = MlflowClient("http://localhost:5000")
   client.transition_model_version_stage("churn-predictor", "2", "Archived")
   ```

2. **Re-promote** the previous version:
   ```bash
   python scripts/promote_model.py --version 1 --force
   ```

3. **Restart the API**:
   ```bash
   docker-compose restart api
   ```

---

## 10. Running Tests

### Integration tests
```bash
cd api
pip install pytest httpx
pytest tests/ -v --tb=short
```

### With coverage report
```bash
pytest tests/ --cov=. --cov-report=html
# Open htmlcov/index.html in your browser
```

### Load testing
```bash
pip install locust

# With web UI (interactive)
locust -f scripts/locustfile.py --host http://localhost:8000
# Open http://localhost:8089

# Headless (automated)
locust -f scripts/locustfile.py --host http://localhost:8000 \
  --users 200 --spawn-rate 10 --run-time 5m --headless
```

### Drift simulation
```bash
python scripts/simulate_drift.py
# Watch Grafana at http://localhost:3000 while running
```

---

## 11. Monitoring

### Grafana Dashboard
1. Open http://localhost:3000 (admin/admin)
2. Navigate to Dashboards → MLflow Pipeline → "Churn Prediction API"
3. The dashboard shows:
   - Request rate (req/s)
   - P95 latency
   - Error rate (%)
   - Churn prediction distribution
   - Probability score histogram
   - Batch size distribution

### Prometheus
- Raw metrics: http://localhost:9090
- API metrics endpoint: http://localhost:8000/metrics

---

## 12. Troubleshooting

### MLflow can't connect to PostgreSQL
```
Error: connection refused to localhost:5432
```
**Fix:** Wait 30 seconds for PostgreSQL to initialize, or check:
```bash
docker-compose logs postgres
```

### MLflow can't upload artifacts to MinIO
```
Error: An error occurred (AccessDenied)
```
**Fix:** Check that the `mlflow` bucket exists:
```bash
docker-compose logs minio-setup
```
If the bucket is missing, recreate it:
```bash
docker-compose restart minio-setup
```

### FastAPI returns 503 "Model not loaded"
**Cause:** The API started before the model was trained/registered.
**Fix:** Train a model first, then restart the API:
```bash
cd training && sbt run
docker-compose restart api
```

### Spark "Java module access" errors
```
WARNING: An illegal reflective access operation has occurred
```
**Fix:** Make sure you're using Java 11 (not 17+). Check: `java -version`

### Docker containers keep restarting
```bash
# Check logs for the failing container
docker-compose logs -f <service-name>

# Nuclear option: clean restart
docker-compose down -v  # WARNING: deletes all data
docker-compose up -d
```

### Port conflicts
If a port is already in use, edit `.env` to change it:
```bash
MLFLOW_PORT=5001     # default: 5000
API_PORT=8001        # default: 8000
GRAFANA_PORT=3001    # default: 3000
```

---

## Common Commands Cheat Sheet

```bash
# Start everything
docker-compose up -d

# Stop (keep data)
docker-compose down

# Stop + delete all data
docker-compose down -v

# View logs
docker-compose logs -f mlflow
docker-compose logs -f api

# Rebuild API after code changes
docker-compose up -d --build api

# Check service status
docker-compose ps

# Run training
cd training && sbt run

# Run tests
cd api && pytest tests/ -v

# Run load test
locust -f scripts/locustfile.py --host http://localhost:8000
```
