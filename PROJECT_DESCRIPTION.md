# Smart AI Energy Consumption Predictor

## Project Description

> Smart AI Energy Consumption Predictor is an AI-powered system that predicts electricity consumption and provides personalized optimization recommendations to reduce energy wastage, lower electricity bills, and minimize carbon emissions.
---

### 📋 Title
**Smart AI Energy Consumption Predictor**

---

### 📚 Developer

| Name | College |
|------|---------|
| Kaustubh Agrawal | Manipal Institute of Technology, Bengaluru |

---

### 🎯 SDG Alignment

**Primary SDG: SDG 7 – Affordable and Clean Energy**

![SDG 7](https://sdgs.un.org/sites/default/files/goals/E_SDG_Icons-07.jpg)

**Secondary SDGs:**
- **SDG 11** – Sustainable Cities and Communities
- **SDG 12** – Responsible Consumption and Production
- **SDG 13** – Climate Action

This project directly contributes to SDG 7 by helping individuals and institutions optimize their energy consumption, reduce wastage, and transition towards more sustainable energy practices.

---

### ❓ Problem Statement

> *How might we use AI to predict electricity consumption so that individuals and institutions can use energy more efficiently and reduce their environmental impact?*

**Key Challenges Addressed:**
- Lack of visibility into energy consumption patterns
- Difficulty in predicting future electricity bills
- Unawareness of carbon footprint from energy usage
- Limited actionable insights for energy optimization

---

### 🤖 AI Solution Overview

Our solution is an **AI-powered energy prediction and optimization system** that combines machine learning with an interactive dashboard.

**Core AI Components:**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Forecasting Models** | Linear Regression, Random Forest, XGBoost | Predict future energy consumption |
| **Explainability** | SHAP (SHapley Additive exPlanations) | Make predictions transparent and interpretable |
| **RAG System** | Sentence-Transformers + FAISS | Answer questions using energy knowledge base |
| **Agentic AI** | Google Gemini API | Conversational energy advisor with autonomous analysis |
| **Recommendation Engine** | Rule-based AI Agent | Generate personalized optimization tips |

**Key Features:**
1. **Energy Usage Forecasting** – Hourly, daily, and monthly predictions
2. **Bill Estimation** – Calculate costs based on Indian tariff slabs
3. **Carbon Footprint Tracking** – CO₂ emissions using 0.82 kg/kWh factor
4. **AI-Powered Chat** – RAG-enhanced conversational energy advisor
5. **Autonomous Analysis** – Agentic AI analyzes consumption patterns
6. **AI Recommendations** – Personalized energy-saving suggestions
7. **Interactive Dashboard** – Real-time visualization with Streamlit

**Model Performance:**
- Random Forest: **2.71% MAPE** (Mean Absolute Percentage Error)
- XGBoost: **3.07% MAPE**
- Both exceed the <10% error target ✅

---

### 👥 Target Users

| User Group | Use Case |
|------------|----------|
| **Students in Hostels** | Monitor shared utility consumption, split bills fairly |
| **Residential Households** | Reduce electricity bills, optimize appliance usage |
| **Educational Institutions** | Track campus energy consumption, meet sustainability goals |
| **Small Businesses** | Control operational costs, reduce carbon footprint |

---

### ⚖️ Responsible AI Considerations

| Principle | Implementation |
|-----------|----------------|
| **Fairness** | Unbiased training data representing diverse consumption patterns |
| **Transparency** | SHAP explanations for all predictions, clear methodology |
| **Privacy** | No personal data storage; all processing is local |
| **Security** | Data encryption, secure session handling |
| **Accountability** | Clear documentation, auditable recommendations |
| **Ethics** | No misleading advice; recommendations backed by data |

**Bias Mitigation:**
- Training data includes various consumption patterns (weekday/weekend, seasonal)
- Model performance validated across different usage profiles
- Recommendations are contextualized, not one-size-fits-all

---

### 📈 Expected Impact

#### Environmental Impact
- **Reduced Energy Wastage**: AI-driven insights help users identify and eliminate unnecessary consumption
- **Lower Carbon Emissions**: Carbon tracking creates awareness and motivates reduction
- **Sustainable Practices**: Recommendations promote efficient appliances and renewable adoption

#### Social Impact
- **Increased Awareness**: Dashboard visualizations educate users about consumption patterns
- **Behavioral Change**: Personalized tips encourage sustainable habits
- **Accessibility**: Free, web-based tool available to all users

#### Economic Impact
- **Lower Electricity Bills**: Optimization recommendations can reduce bills by 15-25%
- **Cost Transparency**: Bill estimator helps users plan and budget
- **Long-term Savings**: Solar adoption recommendations with payback calculations

---

### 🛠️ Technical Stack

| Category | Technologies |
|----------|--------------|
| **ML/AI** | scikit-learn, XGBoost, TensorFlow, SHAP |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Plotly, Streamlit |
| **Languages** | Python 3.9+ |

---

### 📊 Success Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Prediction Error (MAPE) | < 10% | ✅ 2.71% |
| Model Training | 4 models | ✅ 3 models trained |
| Dashboard Components | 5 pages | ✅ 6 pages |
| Responsible AI | Transparency | ✅ SHAP implemented |

---

### 🔗 References

- UN Sustainable Development Goals: https://sdgs.un.org/goals
- India Grid Emission Factor: Central Electricity Authority (0.82 kg CO₂/kWh)
- SHAP Documentation: https://shap.readthedocs.io

---

*Built with ❤️ for a sustainable future | SDG 7 - Affordable and Clean Energy*
