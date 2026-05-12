# ML Production Pipeline with MLflow

> **Project 9** — Master's Semester Project  
> Full MLOps pipeline: Training (Spark/Scala) → Tracking (MLflow) → Serving (FastAPI) → Monitoring (Prometheus/Grafana)

---

## 🏗️ Architecture Overview

```
[CSV Dataset]
     ↓
[Spark Preprocessing Pipeline] (Scala)
     ↓
[Spark MLlib Training] (Scala) ──logs──→ [MLflow Tracking Server]
                                               ↓
                                     [MLflow Model Registry]
                                          (PostgreSQL + MinIO)
                                               ↓
                                    [FastAPI Prediction Service]
                                         ↓            ↓
                                  [/predict]    [/predict/batch]
                                         ↓
                                   [Prometheus] → [Grafana Dashboard]
```

---

## 📋 Prerequisites

Before you start, make sure you have the following installed:

| Tool | Version | Check command |
|---|---|---|
| Docker Desktop | Latest | `docker --version` |
| Docker Compose | v2+ (included in Docker Desktop) | `docker compose version` |
| Java JDK | **11** (not 17!) | `java -version` |
| SBT | Latest | `sbt --version` |
| Python | 3.10+ | `python --version` |
| Git | Latest | `git --version` |

> ⚠️ **Important:** Use Java 11, not 17. Spark 3.4 has compatibility issues with Java 17.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd project9-mlflow
```

### 2. Set up environment variables
```bash
# Copy the template
cp .env.example .env

# Edit .env if you want to change passwords (optional for local dev)
```

### 3. Download the dataset
Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place the CSV file in:
```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 4. Start all services
```bash
docker-compose up -d
```

This will start 9 containers:
- PostgreSQL (MLflow backend)
- MinIO (artifact storage)
- MinIO Setup (creates the bucket — runs once and exits)
- MLflow Tracking Server
- Spark Master + 2 Workers
- FastAPI (prediction API)
- Prometheus (metrics collector)
- Grafana (dashboards)

### 5. Verify all services are running
```bash
docker-compose ps
```

---

## 🌐 Service URLs

Once everything is running, you can access:

| Service | URL | Credentials |
|---|---|---|
| **MLflow UI** | [http://localhost:5000](http://localhost:5000) | No auth required |
| **Spark Master UI** | [http://localhost:8080](http://localhost:8080) | No auth required |
| **FastAPI Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | No auth required |
| **FastAPI Health** | [http://localhost:8000/health](http://localhost:8000/health) | No auth required |
| **MinIO Console** | [http://localhost:9001](http://localhost:9001) | `minioadmin` / `minioadmin_secret_2024` |
| **Prometheus** | [http://localhost:9090](http://localhost:9090) | No auth required |
| **Grafana** | [http://localhost:3000](http://localhost:3000) | `admin` / `admin` |

---

## 🧪 Verify the Setup

### Check MLflow is working
```bash
# Should return a JSON response
curl http://localhost:5000/health
```
Then open [http://localhost:5000](http://localhost:5000) in your browser — you should see the MLflow UI.

### Check MinIO bucket exists
Open [http://localhost:9001](http://localhost:9001), log in, and verify that a bucket called `mlflow` exists.

### Check FastAPI is responding
```bash
curl http://localhost:8000/health
```

### Check Spark cluster
Open [http://localhost:8080](http://localhost:8080) — you should see the Spark Master with 2 workers registered.

### Check Grafana datasource
Open [http://localhost:3000](http://localhost:3000), log in with `admin/admin`, go to **Configuration → Data Sources** and verify Prometheus is connected.

---

## 📁 Project Structure

```
project9-mlflow/
├── docker-compose.yml            ← all services wired together
├── .env                          ← environment variables (never commit!)
├── .env.example                  ← template for .env
├── README.md                     ← this file
├── DEPLOYMENT.md                 ← full deployment & operations guide
│
├── training/                     ← Scala/SBT training project
│   ├── build.sbt
│   ├── project/plugins.sbt
│   └── src/main/scala/
│       ├── DataLoader.scala
│       ├── Preprocessor.scala
│       ├── Trainer.scala
│       ├── ModelRegistry.scala
│       └── Main.scala
│
├── api/                          ← FastAPI prediction service
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── models/schemas.py
│   ├── services/model_service.py
│   ├── routers/predict.py
│   ├── routers/health.py
│   └── tests/
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── datasources.yml
│       └── dashboards/
│
├── scripts/
│   ├── simulate_drift.py
│   ├── compare_models.py
│   └── promote_model.py
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
└── report/
    └── Project9_Report.pdf
```

---

## 🛑 Common Commands

```bash
# Start all services
docker-compose up -d

# Stop all services (keeps data)
docker-compose down

# Stop and DELETE all data (fresh start)
docker-compose down -v

# View logs for a specific service
docker-compose logs -f mlflow
docker-compose logs -f api

# Rebuild the FastAPI container after code changes
docker-compose up -d --build api

# Check which containers are running
docker-compose ps
```

---

## 📊 Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Training | Scala 2.12 + Spark 3.4 MLlib | Distributed ML training |
| Experiment Tracking | MLflow 2.x | Log params, metrics, artifacts |
| Model Registry | MLflow Model Registry | Version and lifecycle management |
| Metadata Storage | PostgreSQL 15 | MLflow backend store |
| Artifact Storage | MinIO (S3-compatible) | Store models, plots, configs |
| Prediction API | FastAPI (Python) | REST API for serving predictions |
| Monitoring | Prometheus + Grafana | Metrics collection and dashboards |
| Containerization | Docker + Docker Compose | Infrastructure as code |
| Testing | pytest + Locust | Integration and load testing |
