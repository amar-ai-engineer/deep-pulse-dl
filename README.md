# DeepPulse AI — Predictive Maintenance with LSTM on NASA C-MAPSS

A deep learning system that predicts how many operational cycles a turbofan engine has left before it needs maintenance. Trained on the NASA C-MAPSS benchmark dataset, it achieves RMSE of 12.6 cycles — 68% better than a naive baseline.

**Live demo:** select any of the 100 test engines and see the LSTM's predicted remaining useful life overlaid against the actual ground truth. No API key or GPU required.

---

## What problem does this solve?

Unplanned engine failures cost airlines $150,000+ per hour of downtime. The traditional approach — maintenance on a fixed schedule — means either servicing engines too early (wasted cost) or too late (risk of failure).

Predictive maintenance uses sensor data to predict exactly when each engine needs attention. You service it at the right time, every time.

---

## Business impact

- A 1% reduction in unplanned downtime on a mid-size fleet saves ~$2M per year
- Extends engine life by avoiding unnecessary early maintenance
- Prioritizes which engines need attention this week vs next month
- Generates data-driven maintenance schedules instead of calendar-based ones

---

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

No API key or GPU required. The app loads pre-computed predictions from `data/predictions/`.

---

## Project structure

```
deep-pulse-dl/
├── app.py                                  # 4-tab Streamlit dashboard
├── src/
│   ├── data_pipeline.py                    # NASA C-MAPSS loader + feature engineering
│   ├── predictor.py                        # PredictionLoader for pre-computed results
│   ├── analytics.py                        # Fleet health, maintenance scheduling
│   └── visualization.py                    # Plotly chart builders
├── data/
│   ├── nasa-dataset/                       # NASA C-MAPSS FD001–FD004 raw files
│   └── predictions/
│       ├── lstm_predictions_FD001.csv      # Pre-computed LSTM predictions (13,096 rows)
│       ├── training_history.json           # Loss curves (50 epochs)
│       └── model_evaluation.json           # RMSE, MAE, R², baseline comparison
├── notebooks/
│   └── lstm_training.ipynb                 # Full LSTM training pipeline (Colab)
└── requirements.txt
```

---

## The 4 tabs

**Fleet Overview** — all 100 test engines plotted by predicted RUL, color-coded Critical/Warning/Healthy. Shows which engines need attention this week.

**Engine Deep Dive** — select any engine to see actual vs predicted RUL over its full cycle history, normalized sensor trends, and a maintenance recommendation.

**Model Performance** — training loss curves, actual vs predicted scatter plot, error distribution, and comparison to the naive mean baseline.

**How It Works** — RUL prediction explained, LSTM architecture, why we cap RUL at 125, feature engineering decisions.

---

## Model details

| Config | Value |
|--------|-------|
| Architecture | LSTM (2 layers, 128 → 64 units) |
| Input | 30-cycle sliding window, 14 sensors |
| RUL cap | 125 cycles |
| Training epochs | 50 |
| RMSE | 12.6 cycles |
| MAE | 10.1 cycles |
| R² | 0.90 |
| Baseline RMSE | 40.1 cycles |

---

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) FD001:
- 100 training engines, 100 test engines
- Single operating condition, one fault mode
- 21 sensors per cycle (14 useful after removing flat sensors)

---

Built by **Amar Ismail** | [GitHub](https://github.com/amar-ai-engineer) | [LinkedIn](https://www.linkedin.com/in/amar-ai-engineer/)
