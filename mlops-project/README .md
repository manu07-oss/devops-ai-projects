# 🚀 MLOps Salary Prediction Pipeline

> **End-to-end MLOps project** — From raw data to a monitored, versioned, production-deployed ML model using industry-standard DevOps + ML tooling.

---

## 📌 What Problem Does This Solve?

In traditional ML, a data scientist trains a model on their laptop and hands over a `.pkl` file. Nobody knows:
- Which version of the model is in production
- What data it was trained on
- How it's performing today vs last week
- How to redeploy it when it degrades

**This project solves all of that** by wrapping an ML model with a full DevOps lifecycle — versioning, CI/CD, containerization, deployment, and monitoring — exactly how it's done in production ML teams.

---

## 🎯 Use Case

**Predict employee salary** based on years of experience and education level.

Simple model. Complex infrastructure around it. That's the point.

| Input | Output |
|-------|--------|
| Years of Experience | Predicted Salary (USD) |
| Education Level (1–4) | |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DEVELOPER LAPTOP                         │
│                                                                 │
│   data/  ──► src/train.py ──► MLflow Tracking ──► Model Registry│
│   (DVC)       (scikit-learn)    (Experiments)      (Versioned)  │
└──────────────────────────┬──────────────────────────────────────┘
                           │  git push
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CI/CD PIPELINE                             │
│                   (GitHub Actions)                              │
│                                                                 │
│   Lint ──► Test ──► Build Docker Image ──► Push to Registry    │
└──────────────────────────┬──────────────────────────────────────┘
                           │  deploy
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PRODUCTION (Kubernetes)                       │
│                                                                 │
│   FastAPI Service  ──►  /predict endpoint                      │
│   (Docker Container)    POST { experience, education }         │
│                          → { predicted_salary: 85000 }         │
└──────────────────────────┬──────────────────────────────────────┘
                           │  metrics
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MONITORING STACK                            │
│                                                                 │
│   Prometheus ──► Grafana Dashboard ──► Drift Alerts            │
│   (scrape metrics)  (visualize)        (Evidently AI)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧱 Tech Stack — What It Is & Why

| Layer | Tool | What It Does | Why This Tool |
|-------|------|-------------|---------------|
| **ML Training** | scikit-learn | Trains the LinearRegression model | Industry standard ML library |
| **Experiment Tracking** | MLflow | Logs every training run — params, metrics, model artifacts | Like Git, but for ML experiments |
| **Data Versioning** | DVC | Versions the dataset alongside code | So you know exactly which data produced which model |
| **Model Serving** | FastAPI | Exposes model as a REST API `/predict` | Lightweight, production-grade Python API framework |
| **Containerization** | Docker | Packages the API + model into a portable image | Runs identically on any machine or cloud |
| **CI/CD** | GitHub Actions | Auto-tests and deploys on every `git push` | Automates the entire release pipeline |
| **Orchestration** | Kubernetes (EKS) | Runs and scales the Docker containers | Industry standard container orchestration |
| **Monitoring** | Prometheus + Grafana | Tracks API latency, request count, error rate | Same stack used in production SRE teams |
| **Drift Detection** | Evidently AI | Alerts when model predictions degrade over time | Catches model decay before it affects users |

---

## 📦 What Needs to Be Installed

### Local Machine (Windows + VS Code)
```bash
# Python 3.9+
python --version

# Core Python packages
pip install pandas scikit-learn mlflow fastapi uvicorn dvc evidently

# Docker Desktop
# Download: https://www.docker.com/products/docker-desktop

# kubectl (for Kubernetes)
# Download: https://kubernetes.io/docs/tasks/tools/
```

### Cloud / Infrastructure
- AWS account (for EKS deployment in Week 3)
- GitHub account (for Actions CI/CD)

---

## 🗂️ Project Structure

```
mlops-project/
│
├── data/                        # Dataset (versioned with DVC)
│   └── salary_data.csv
│
├── src/
│   ├── train.py                 # Model training + MLflow logging
│   ├── predict.py               # FastAPI prediction service
│   └── monitor.py               # Drift detection with Evidently
│
├── docker/
│   └── Dockerfile               # Container definition
│
├── k8s/
│   ├── deployment.yaml          # Kubernetes deployment
│   └── service.yaml             # Kubernetes service (LoadBalancer)
│
├── .github/
│   └── workflows/
│       └── ci-cd.yaml           # GitHub Actions pipeline
│
├── monitoring/
│   ├── prometheus.yaml          # Metrics scraping config
│   └── grafana-dashboard.json   # Pre-built dashboard
│
├── mlflow.db                    # Local MLflow tracking store
├── .dvc/                        # DVC configuration
├── .gitignore
└── README.md
```

---

## 🔄 How It Works — Step by Step

### Step 1: Train & Track
```bash
python src/train.py
# → Trains model
# → Logs MAE, R2 to MLflow
# → Saves versioned model artifact
```

### Step 2: Serve as API
```bash
uvicorn src.predict:app --reload
# → API running at http://localhost:8000/predict
```

```bash
# Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"years_experience": 5, "education_level": 3}'

# Response:
# { "predicted_salary": 72450.0 }
```

### Step 3: Containerize
```bash
docker build -t mlops-salary-api:v1 .
docker run -p 8000:8000 mlops-salary-api:v1
```

### Step 4: CI/CD on Push
```bash
git push origin main
# → GitHub Actions triggers
# → Runs tests
# → Builds Docker image
# → Deploys to Kubernetes
```

### Step 5: Monitor in Production
- Prometheus scrapes `/metrics` endpoint every 15 seconds
- Grafana shows real-time dashboard
- Evidently detects if prediction distribution shifts (model drift)

---

## 📊 MLflow Experiment Tracking

Every training run is logged:

| Run | Model Type | MAE | R2 Score | Date |
|-----|-----------|-----|---------|------|
| trusting-jay-765 | LinearRegression | 16,536 | 0.61 | Day 1 |
| ... | RandomForest | TBD | TBD | Day 2 |
| ... | XGBoost | TBD | TBD | Day 2 |

MLflow UI: `http://localhost:5000`

---

## 🧠 Is This Frontend, Backend, or ML?

**This project spans all three layers:**

```
Frontend Layer:   Grafana Dashboard (visualizes model health)
                  MLflow UI (visualizes experiments)

Backend Layer:    FastAPI REST API (serves predictions)
                  Kubernetes (orchestrates containers)
                  GitHub Actions (automates deployments)

ML Layer:         scikit-learn model (the actual prediction logic)
                  MLflow (experiment tracking)
                  DVC (data versioning)
                  Evidently (drift detection)
```

**The DevOps/MLOps layer sits between backend and ML** — it's the infrastructure that makes ML reliable, reproducible, and scalable in production.

---

## 💼 Interview Explanation

### "Tell me about this project"

> "I built an end-to-end MLOps pipeline that takes a machine learning model from training to production. The core idea was to treat ML models like software — with versioning, CI/CD, containerization, and monitoring. I used MLflow to track every experiment, DVC to version the training data, Docker to containerize the prediction API, GitHub Actions to automate deployments to Kubernetes, and Prometheus with Grafana to monitor model health in production. The business problem it solves is making ML deployments reliable and reproducible — any team member can retrain, redeploy, and monitor the model without manual steps."

### "What is MLflow?"

> "MLflow is an open-source platform for managing the ML lifecycle. I used it to log every training run — the parameters I used, the metrics like MAE and R2 score, and the trained model artifact. It's like Git for experiments — you can compare 10 different runs side by side and promote the best one to production."

### "What is model drift?"

> "Model drift happens when real-world data starts looking different from the data the model was trained on — so predictions become less accurate over time. I used Evidently AI to detect this by comparing the distribution of incoming prediction requests against the training data distribution. When drift is detected, it triggers an alert so the model can be retrained."

### "Why Kubernetes for this?"

> "The prediction API needs to be scalable and highly available. Kubernetes lets me define the desired state — say, 3 replicas of the prediction service — and it handles restarts, scaling, and rolling deployments automatically. It also integrates with my CI/CD pipeline so every code push can trigger a zero-downtime deployment."

---

## 📈 Outcomes & What You Learn

By completing this project you demonstrate:

- ✅ **MLOps fundamentals** — experiment tracking, model versioning, reproducibility
- ✅ **ML in production** — serving models as REST APIs, not just Jupyter notebooks
- ✅ **DevOps applied to ML** — CI/CD, Docker, Kubernetes for ML workloads
- ✅ **Observability** — monitoring ML systems with Prometheus, Grafana, drift detection
- ✅ **End-to-end ownership** — from raw data to production API to alerting

---

## 🚧 Project Roadmap

- [x] Week 1 — Model training + MLflow experiment tracking + DVC
- [ ] Week 2 — FastAPI serving + Docker + GitHub Actions CI/CD
- [ ] Week 3 — Kubernetes (EKS) deployment + Argo Workflows (auto-retrain)
- [ ] Week 4 — Prometheus + Grafana monitoring + Evidently drift detection

---

## 👩‍💻 Author

**Manogna** — DevOps Engineer  
🔗 [GitHub](https://github.com/manu07-oss) | [LinkedIn](#)

> *"Infrastructure for ML is still infrastructure. DevOps engineers who understand both worlds are rare — and in demand."*

---

## 📄 License

MIT License — free to use and learn from.
