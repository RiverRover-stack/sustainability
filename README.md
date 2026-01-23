# ⚡ Smart AI Energy Consumption Predictor

> AI-powered system for predicting electricity consumption and providing personalized optimization recommendations using **RAG** and **Agentic AI**.

**🎯 SDG 7 - Affordable and Clean Energy**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![Gemini](https://img.shields.io/badge/Gemini-AI-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **📊 Energy Usage Forecasting** - Predict consumption using ML models (Linear Regression, Random Forest, XGBoost)
- **🤖 RAG-Powered Chat** - Ask questions using knowledge base with semantic search
- **🧠 Agentic AI** - Google Gemini-powered conversational energy advisor
- **💰 Bill Estimation** - Calculate electricity bills based on Indian tariff slabs
- **🌍 Carbon Tracking** - Monitor CO₂ emissions with India-specific emission factors
- **💡 AI Recommendations** - Personalized energy optimization tips
- **🔍 SHAP Explainability** - Transparent, interpretable predictions

## 📁 Project Structure

```
sustainability/
├── data/
│   └── energy_data.csv          # Synthetic dataset (1 year hourly)
├── models/
│   ├── linear_regression.pkl    # Baseline model
│   ├── random_forest.pkl        # Ensemble model
│   ├── xgboost_model.pkl        # Gradient boosting
│   ├── model_comparison.csv     # Performance metrics
│   └── feature_importance.csv   # SHAP feature rankings
├── src/
│   ├── data_generator.py        # Synthetic data generation
│   ├── preprocessing.py         # Feature engineering
│   ├── train_models.py          # ML training pipeline
│   ├── recommender.py           # AI recommendation engine
│   ├── carbon_calculator.py     # CO₂ emission calculations
│   ├── knowledge_base.py        # RAG knowledge base (NEW)
│   └── energy_agent.py          # Agentic AI advisor (NEW)
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd d:\sustainability
pip install -r requirements.txt
```

### 2. Generate Data & Train Models

```bash
# Generate synthetic energy data
python src/data_generator.py

# Train all models
python src/train_models.py
```

### 3. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

### 4. Configure AI Chat (Optional)

1. Get a free API key from https://makersuite.google.com/app/apikey
2. Go to **🤖 AI Assistant** page
3. Enter API key in sidebar settings

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 📈 Overview | Consumption graphs, peak analysis, daily patterns |
| 🔮 Forecasting | Model predictions with SHAP explanations |
| 💰 Bill Estimator | Calculate bills using tariff slabs |
| 🌍 Carbon Footprint | CO₂ tracking and reduction scenarios |
| 💡 Recommendations | Personalized optimization tips |
| 🤖 AI Assistant | RAG chat, auto-analysis, knowledge search |
| 📤 Data Upload | Upload your own consumption data |

## 🤖 AI Technologies

| Technology | Component | Purpose |
|------------|-----------|---------|
| **RAG** | Sentence-Transformers + FAISS | Semantic search on energy knowledge |
| **Agentic AI** | Google Gemini API | Conversational energy advisor |
| **ML Models** | Random Forest, XGBoost | Consumption forecasting |
| **XAI** | SHAP | Model explainability |

### RAG Knowledge Base Topics
- Energy saving tips (AC, LED, off-peak, standby)
- Solar energy & PM Surya Ghar scheme
- Carbon footprint & emission factors
- Appliance efficiency (BEE star ratings)
- Electricity tariffs in India

## 🤖 ML Models

| Model | MAPE | Description |
|-------|------|-------------|
| Linear Regression | 0.00% | Baseline reference |
| Random Forest | **2.71%** | Best performer |
| XGBoost | 3.07% | Gradient boosting |

**Target**: MAPE < 10% ✅

## 🌱 Carbon Calculation

```
CO₂ (kg) = kWh × 0.82
```
*India grid emission factor: 0.82 kg CO₂/kWh*

## 🎯 Responsible AI Principles

- ✅ **Fairness** - Unbiased datasets and recommendations
- ✅ **Transparency** - SHAP explanations for predictions
- ✅ **Privacy** - No personal data storage
- ✅ **Ethics** - No misleading advice

## 📈 Expected Impact

| Dimension | Impact |
|-----------|--------|
| 🌍 Environmental | Reduced energy wastage, lower emissions |
| 👥 Social | Increased sustainability awareness |
| 💰 Economic | Lower electricity bills (15-25% savings) |

## 📄 License

MIT License - Built for SDG 7: Affordable and Clean Energy

---

**Developer**: Kaustubh Agrawal | Manipal Institute of Technology, Bengaluru

*Built with ❤️ for a sustainable future*
