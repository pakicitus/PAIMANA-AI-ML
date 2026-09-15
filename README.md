# PAIMANA AI

## AI-Powered Predictive Infrastructure Project Monitoring & Risk Intelligence

> **From Monitoring Projects to Predicting Risk**

PAIMANA AI is an AI-powered predictive risk intelligence system designed to enhance infrastructure project monitoring under the PAIMANA (Project Appraisal, Monitoring and Information System for Infrastructure and Macro-Economy) ecosystem.

Traditional monitoring mainly describes the current status of projects. PAIMANA AI adds a predictive layer that asks: **What is likely to go wrong next?**

The system analyses historical and current project information to identify emerging cost, schedule and implementation risks, converts those signals into a simple 0–100 risk score, and explains the major factors behind the prediction.

## 🎯 Problem

Large infrastructure projects involve significant public investment and can span several years. During implementation they may experience:

- Cost escalation
- Commissioning delays
- Slow physical progress
- Increasing expenditure
- Milestone slippage
- Financial–physical progress mismatches
- Repeated cost and schedule revisions

The goal is to move monitoring from primarily **descriptive and retrospective** analysis toward **predictive and proactive** risk intelligence.

## 💡 Solution

PAIMANA AI acts as an intelligence layer over existing project-monitoring information. It is designed to use information already available through monitoring workflows rather than requiring a completely separate reporting process.

It can:

1. Predict potential cost risk
2. Predict potential schedule risk
3. Provide an additional time-to-risk signal
4. Generate a unified project risk score
5. Categorize projects by risk level
6. Explain major risk drivers
7. Help prioritize projects for closer attention
8. Support portfolio, sector, state and agency-level analysis

## 🚀 Key Capabilities

### 1. Predictive Cost Risk

The system estimates whether a project is showing patterns associated with future cost deterioration. This moves beyond simply reporting an already-realized cost increase and provides an early-warning signal.

### 2. Predictive Schedule Risk

A separate prediction identifies projects showing patterns associated with future schedule or time deterioration.

A project can therefore have low cost risk but high schedule risk, high cost risk but low schedule risk, high risk in both dimensions, or low risk overall.

### 3. Unified 0–100 Project Risk Score

Three risk signals are combined:

- **40% Cost Risk**
- **40% Schedule Risk**
- **20% Survival / Time-to-Risk Signal**

Risk levels:

| Score | Risk Level |
|---:|---|
| 0–24.99 | LOW |
| 25–49.99 | WATCH |
| 50–74.99 | ELEVATED |
| 75–100 | CRITICAL |

This gives monitoring authorities a simple way to prioritize a large project portfolio.

### 4. Explainable Risk Analysis

The system does not only say that a project is risky. It can identify the factors that contributed most strongly to the prediction using SHAP-based explainability.

Example:

```text
Project Risk: CRITICAL
Risk Score: 86.5 / 100

Major Risk Drivers:
• Schedule deterioration
• Increasing cost trend
• Financial–physical progress mismatch
• Historical agency risk
• Project timeline deterioration
```

These are model contributions, not proof of causal relationships.

### 5. Project Trend Analysis

Projects are dynamic, so the system considers how conditions change over time, including:

- Cost trends
- Expenditure trends
- Schedule changes
- Milestone progress
- Cost revisions
- Progress mismatches

### 6. Financial Progress Monitoring

The system uses:

- Original project cost
- Revised project cost
- Anticipated project cost
- Cumulative expenditure

to identify significant financial changes and unusual expenditure patterns.

### 7. Physical / Milestone Progress Monitoring

Milestone information is used to understand implementation progress and compare it with financial progress and project timelines.

### 8. Financial–Physical Progress Mismatch

The system can identify situations such as:

```text
High expenditure
      +
Low milestone completion
      +
Schedule progression
      ↓
Potential Risk Signal
```

### 9. Agency Risk Intelligence

Historical project outcomes provide an agency-level risk signal. This gives context to an individual project based on historical patterns without automatically labeling an agency as good or bad.

### 10. Sector Risk Intelligence

Sector-level historical patterns provide context for individual projects and enable questions such as:

- Which sectors contain more high-risk projects?
- How does a project compare with historical sector behaviour?
- Which sectors show higher historical overrun patterns?

### 11. State and Portfolio Analysis

Projects can be aggregated by:

- State
- Sector
- Implementing agency
- Risk level
- Project status

This enables a drill-down from portfolio to individual project and then to its risk drivers.

## 🧠 Machine Learning Architecture

The system uses multiple complementary models:

- **XGBoost classifier for cost risk**
- **XGBoost classifier for schedule risk**
- **Cox Proportional Hazards model for an additional time-to-risk signal**
- **SHAP for explainability**

```text
                         Project
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
          XGBoost       XGBoost         Cox
            Cost        Schedule       Survival
            Risk          Risk          Risk
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                       Risk Fusion
                            ▼
                     Risk Score 0–100
                            ▼
              LOW / WATCH / ELEVATED / CRITICAL
```

## 📊 Model Performance

The models were evaluated using chronological train/test evaluation.

### Cost Risk Model

| Metric | Result |
|---|---:|
| PR-AUC | **0.4368** |
| ROC-AUC | **0.7932** |
| F1 Score | **0.4566** |
| Precision | **0.5865** |
| Recall | **0.3738** |

### Schedule Risk Model

| Metric | Result |
|---|---:|
| PR-AUC | **0.8091** |
| ROC-AUC | **0.8954** |
| F1 Score | **0.7211** |
| Precision | **0.7387** |
| Recall | **0.7044** |

### Survival Model

The Cox survival model provides a supplementary time-to-risk signal and achieved a concordance index of approximately **0.586** on the project-level survival dataset.

## 🧩 Information Used by the Models

The engineered inputs represent several broad dimensions of project behaviour.

### Project Characteristics
- Project age
- Original duration
- Original cost
- Current cost
- Sector
- State
- Approval information

### Financial Indicators
- Cumulative expenditure
- Cost revision
- Anticipated cost
- Expenditure relative to project cost
- Cost growth trends
- Expenditure growth trends

### Schedule Indicators
- Anticipated delay
- Remaining original duration
- Schedule changes
- Duration overrun
- Project duration progression

### Milestone Indicators
- Milestones achieved
- Total milestones
- Milestone completion
- Change in milestone progress

### Progress Mismatch Indicators
- Expenditure versus milestone progress
- Financial versus physical progress gap
- Schedule versus milestone progress mismatch

### Historical Benchmark Indicators
- Agency historical overrun rate
- Sector historical overrun rate
- Historical agency project count

## 🔄 End-to-End Prediction Flow

```text
Raw PAIMANA Project Data
            │
            ▼
    Feature Engineering
            │
            ▼
 Engineered Model Inputs
            │
      ┌─────┴─────┐
      ▼           ▼
Cost XGBoost   Schedule XGBoost
      │           │
      ▼           ▼
 Cost Risk    Schedule Risk
      │           │
      └─────┬─────┘
            ▼
        Cox Model
            │
            ▼
     Survival Signal
            │
            ▼
       Risk Fusion
            │
            ▼
      Risk Score 0–100
            │
            ▼
       Risk Category
            │
            ▼
      Risk Explanation
```

## 📦 Model Package

The repository contains the trained inference package:

```text
predictor.py

cost_final_model.joblib
schedule_final_model.joblib
cox_model.joblib

feature_columns.joblib
cox_feature_columns.joblib

model_manifest.json
model_metadata.json
```

`predictor.py` is the inference engine. It takes the required engineered model inputs, runs the trained models, combines their outputs, and returns the final risk assessment.

Example:

```python
from predictor import predict_project_row

result = predict_project_row(project_features)
print(result)
```

The result contains information such as:

- Cost risk probability
- Schedule risk probability
- Cox/survival risk
- Normalized survival risk
- Composite risk score
- Risk tier
- Model version

## ⚠️ Feature Engineering vs Predictor

An important deployment distinction is:

```text
PAIMANA Database
       │
       ▼
Feature Engineering Layer
       │
       ▼
Engineered Model Inputs
       │
       ▼
predictor.py
       │
       ▼
Risk Prediction
```

`predictor.py` is the **inference engine**. It does not independently reconstruct every historical feature from raw database records.

A production implementation should therefore have a feature-engineering layer that produces the required model inputs consistently with training.

## 🏛️ Proposed PAIMANA Integration

PAIMANA AI is designed as an additional intelligence layer rather than a replacement for the existing monitoring system.

```text
Existing PAIMANA
       │
       ▼
Existing Project Information
       │
       ▼
PAIMANA AI Intelligence Layer
       │
       ├── Risk Prediction
       ├── Early Warning
       ├── Explainability
       ├── Benchmarking
       └── Portfolio Intelligence
       │
       ▼
Monitoring Authorities
       │
       ▼
Earlier Investigation & Intervention
```

The concept is to leverage existing monitoring information and minimize additional manual data-entry requirements.

## 🏗️ Feasibility

The approach is technically feasible because it can work as an intelligence layer over existing project-monitoring information.

```text
Existing Data
     ↓
Existing Monitoring Workflow
     ↓
AI Processing
     ↓
Risk Intelligence
```

A future production deployment could connect the feature-engineering layer to an operational database and expose the predictor through a backend/API service. The current repository should be understood as the ML/inference component, not as proof of a live government deployment.

## 🎯 Expected Impact

PAIMANA AI is intended to provide:

### Earlier Risk Identification
Potentially problematic projects can be highlighted before risks become severe.

### Better Prioritization
Monitoring teams can focus attention on projects with higher predicted risk.

### Faster Analysis
Automated analysis can reduce the effort required to manually examine large numbers of project records.

### Better Portfolio Visibility
Risk patterns can be viewed across projects, sectors, states and agencies.

### Better Decision Support
Risk scores and explanations can help officials decide which projects may require deeper investigation.

### Public Investment Protection
Earlier identification of cost and schedule risks can potentially support earlier corrective action and reduce avoidable escalation.

### Important Note on Savings

The exact amount of money or time saved cannot responsibly be claimed without a live pilot and comparison against the existing monitoring workflow.

The project's impact proposition is:

> **Earlier detection → Earlier intervention → Potentially lower cost and time impact.**

Actual savings can be quantified after deployment using historical back-testing and operational pilot data.

## 💼 Example Use Case

Suppose a large infrastructure project shows:

- Increasing anticipated cost
- Increasing expenditure
- Slow milestone completion
- Schedule deterioration
- A significant financial–physical progress mismatch

A traditional monitoring system may display these indicators separately.

PAIMANA AI can combine them into an alert such as:

```text
┌─────────────────────────────────────┐
│          PROJECT RISK ALERT         │
├─────────────────────────────────────┤
│ Risk Score: 86.5 / 100              │
│ Risk Level: CRITICAL                │
│                                     │
│ Cost Risk:     HIGH                 │
│ Schedule Risk: HIGH                 │
│                                     │
│ Major Risk Drivers:                 │
│ • Schedule deterioration            │
│ • Cost growth                       │
│ • Progress mismatch                 │
│ • Historical risk signals           │
└─────────────────────────────────────┘
```

The project can then be prioritized for closer examination.

## 🔐 Data Privacy

The training datasets are intentionally excluded from this repository.

The `.gitignore` excludes:

```text
*.csv
*.parquet
*.zip
```

This keeps the cleaned and processed project datasets separate from the publicly shareable ML implementation.

The repository contains the model code and artifacts without exposing the underlying training datasets.

## 📁 Repository Structure

```text
PAIMANA-AI-ML/
│
├── predictor.py
│
├── cost_final_model.joblib
├── schedule_final_model.joblib
├── cox_model.joblib
├── feature_columns.joblib
├── cox_feature_columns.joblib
│
├── model_manifest.json
├── model_metadata.json
│
├── pyproject.toml
├── uv.lock
│
├── 02_baselines.ipynb
├── 02_baselines2.ipynb
├── ML_MODEL_PACKAGE.ipynb
├── Phase_4_Explainability.ipynb
├── Phase_5_Risk_Score_Survival.ipynb
├── Phase_6_Model_Packaging.ipynb
├── phase-0.ipynb
│
├── ML_model_files/
│
└── .gitignore
```

Training datasets are intentionally not part of the repository.

## 🔮 Future Enhancements

The current ML system provides the foundation for a broader PAIMANA AI platform.

Potential future capabilities include:

- Real-time project risk monitoring
- Automated early-warning alerts
- Natural-language AI assistant
- Automated project risk reports
- Similar-project comparison
- Advanced sector and agency benchmarking
- Portfolio-level recommendations
- Intervention recommendation engine
- Live PAIMANA integration
- Role-based dashboards
- Continuous model retraining
- Automated monitoring summaries
- Risk trend forecasting

## 🌐 Long-Term Vision

The long-term goal is to evolve infrastructure monitoring from:

```text
DESCRIPTIVE
     │
     ▼
What happened?
     │
     ▼
PREDICTIVE
     │
     ▼
What is likely to happen?
     │
     ▼
PROACTIVE
     │
     ▼
Which projects require attention?
     │
     ▼
What action should be considered?
```

PAIMANA AI is intended to help monitoring authorities move from simply observing project performance toward identifying emerging risks earlier and prioritizing attention.

## 🏆 Project Vision

> **PAIMANA AI — From Monitoring Projects to Predicting Risk.**

The system combines predictive modelling, project trend analysis, survival analysis and explainable AI to create a practical early-warning mechanism for infrastructure project monitoring.

The objective is not to replace human decision-making.

It is to help decision-makers answer three questions faster:

1. **Which projects are at risk?**
2. **Why are they at risk?**
3. **Which projects should receive attention first?**

## 📜 Disclaimer

This project is a prototype developed as part of the **Smart India Hackathon (SIH)** and demonstrates the feasibility of applying machine learning and explainable AI to infrastructure project monitoring.

Predictions are intended to support human decision-making and should not be treated as automatic administrative decisions.

Production deployment would require validation using official operational data, integration with existing government systems, security and privacy controls, domain-expert review, governance processes and continuous performance monitoring.

---

## PAIMANA AI

**Predict. Prioritize. Prevent.**

> **Turning infrastructure monitoring into proactive risk intelligence.**
