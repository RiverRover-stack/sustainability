# ⚡ Smart AI Energy Consumption Predictor

> AI-powered system for predicting electricity consumption and providing personalized optimization recommendations using **RAG**, **Agentic AI**, and **MLOps**.

**🎯 SDG 7 - Affordable and Clean Energy**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.9+-purple.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🌟 Features

### Core ML
- **📊 Energy Forecasting** - ML models (Linear Regression, Random Forest, XGBoost)
- **🔍 SHAP Explainability** - Transparent, interpretable predictions
- **🚨 Anomaly Detection** - Identify unusual consumption patterns

### AI/LLM
- **🤖 RAG-Powered Chat** - Knowledge base with semantic search
- **🧠 Agentic AI** - Google Gemini-powered energy advisor
- **💡 Smart Recommendations** - Personalized optimization tips

### Business Logic
- **💰 Bill Estimation** - Indian tariff slab calculations
- **🌍 Carbon Tracking** - CO₂ emissions with India factors (0.82 kg/kWh)

### MLOps & Production
- **📈 MLflow Tracking** - Experiment logging and model registry
- **⚙️ Optuna Tuning** - Automated hyperparameter optimization
- **🐳 Docker Ready** - Containerized deployment
- **🚀 FastAPI Backend** - REST API for integrations

---

## 📁 Project Structure

```
sustainability/
├── api/                     # FastAPI backend
│   ├── main.py              # API entry point
│   ├── schemas.py           # Pydantic models
│   └── routes/              # Endpoint handlers
├── src/
│   ├── data/                # Data pipeline
│   ├── training/            # ML model trainers
│   ├── mlops/               # MLflow + Optuna
│   ├── anomaly/             # Anomaly detection
│   ├── agent/               # LLM/RAG components
│   ├── carbon/              # Carbon calculations
│   └── train_models.py      # Training orchestrator
├── dashboard/
│   └── app.py               # Streamlit UI
├── Dockerfile               # Container build
├── docker-compose.yml       # Multi-service setup
├── docs/                    # Documentation
│   ├── DECISION_LOG.md      # Technical decisions
│   ├── MODULE_MAP.md        # Module responsibilities
│   └── KNOWN_UNKNOWNS.md    # Limitations
└── ARCHITECTURE.md          # System design
```

---

## 🚀 Quick Start

### Option 1: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Generate data & train models
python src/data/data_generator.py
python src/train_models.py

# Launch dashboard
streamlit run dashboard/app.py
```

### Option 2: Docker

```bash
# Run dashboard only
docker-compose up -d dashboard

# Run with MLflow
docker-compose --profile mlops up -d

# Run with API
docker-compose --profile api up -d
```

### Option 3: API Server

```bash
uvicorn api.main:app --reload
# Swagger docs: http://localhost:8000/docs
```

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict` | POST | Get consumption forecast |
| `/api/v1/recommend` | GET | Get optimization tips |
| `/api/v1/carbon` | POST | Calculate CO₂ emissions |
| `/api/v1/carbon/quick` | GET | Quick carbon lookup |
| `/health` | GET | Health check |

---

## 🤖 ML Models

| Model | MAPE | Status |
|-------|------|--------|
| Linear Regression | 0.00% | ✅ Baseline |
| Random Forest | 2.71% | ✅ Production |
| XGBoost | 3.07% | ✅ Production |
| LSTM | - | 🔄 Planned |

**Target**: MAPE < 10% ✅ Achieved

---

## 📈 MLOps Features

### Experiment Tracking
```bash
# View experiments
mlflow ui --port 5000
```

### Hyperparameter Tuning
```bash
# Optimize XGBoost
python src/mlops/optuna_tuning.py --model xgboost --n-trials 50
```

### Anomaly Detection
```bash
python src/anomaly/detector.py
# Output: 438/8760 anomalies (5.0%)
```

---

## 🌱 Carbon Calculation

```
CO₂ (kg) = kWh × 0.82
```
*India grid emission factor: 0.82 kg CO₂/kWh (CEA 2023)*

---

## 🎯 Responsible AI

- ✅ **Fairness** - Unbiased datasets and recommendations
- ✅ **Transparency** - SHAP explanations for all predictions
- ✅ **Privacy** - No personal data storage
- ✅ **Auditability** - Rule-based recommendations, not black-box

---

## 📈 Expected Impact

| Dimension | Impact |
|-----------|--------|
| 🌍 Environmental | 15-25% energy reduction potential |
| 👥 Social | Increased sustainability awareness |
| 💰 Economic | Lower electricity bills |

---

## 📄 License

MIT License - Built for SDG 7: Affordable and Clean Energy

---

**Developer**: Kaustubh Agrawal | Manipal Institute of Technology, Bengaluru

*Built with ❤️ for a sustainable future*
