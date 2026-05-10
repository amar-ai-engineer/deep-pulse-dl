import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd

from src.data_pipeline import load_cmapss
from src.predictor import PredictionLoader
from src.analytics import compute_fleet_health, compare_to_baseline, generate_maintenance_schedule
from src.visualization import (
    create_fleet_grid,
    create_rul_comparison,
    create_sensor_dashboard,
    create_training_curves,
    create_prediction_scatter,
    create_error_histogram,
)

st.set_page_config(
    page_title="DeepPulse AI - Engine Health Intelligence",
    page_icon=None,
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp { background-color: #f3f4f6; font-family: 'Inter', sans-serif; }
h1 { font-weight: 800 !important; color: #0f172a !important; letter-spacing: -0.025em; }
.kpi-card {
    background: #ffffff; border-radius: 12px; padding: 20px;
    text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    border: 1px solid #e5e7eb;
}
.kpi-value { font-size: 1.8rem; font-weight: 800; color: #0f172a; margin-bottom: 2px; }
.kpi-label { font-size: 0.72rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
.section-card {
    background: #ffffff; border-radius: 12px; padding: 20px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05); border: 1px solid #e5e7eb;
    margin-bottom: 16px;
}
.rec-box {
    background: #f0fdf4; border-left: 4px solid #22c55e;
    padding: 12px 16px; border-radius: 8px; font-size: 0.9rem;
    color: #14532d;
}
.rec-box.warning { background: #fefce8; border-color: #f59e0b; color: #78350f; }
.rec-box.critical { background: #fef2f2; border-color: #ef4444; color: #991b1b; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_data():
    cmapss = load_cmapss("FD001", "data/nasa-dataset")
    loader = PredictionLoader()
    fleet  = loader.get_fleet_status()
    health = compute_fleet_health(fleet)
    all_preds = loader.get_all_predictions()
    eval_meta = loader.get_evaluation()
    history   = loader.get_training_history()
    return cmapss, fleet, health, all_preds, eval_meta, history


cmapss, fleet_df, health, all_preds, eval_meta, history = load_all_data()
loader = PredictionLoader()

STATUS_COLORS = {"Critical": "#ef4444", "Warning": "#f59e0b", "Healthy": "#22c55e"}


def kpi(value, label, color="#0f172a"):
    return f'<div class="kpi-card"><div class="kpi-value" style="color:{color}">{value}</div><div class="kpi-label">{label}</div></div>'


st.markdown("# DeepPulse AI")
st.markdown("*Predictive maintenance for turbofan engines - LSTM on NASA C-MAPSS*")

tab1, tab2, tab3, tab4 = st.tabs([
    "Fleet Overview",
    "Engine Deep Dive",
    "Model Performance",
    "How It Works",
])


# ── Tab 1: Fleet Overview ─────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi(health["total_engines"], "Fleet Size"), unsafe_allow_html=True)
    c2.markdown(kpi(health["critical"], "Critical Engines", "#ef4444"), unsafe_allow_html=True)
    c3.markdown(kpi(f"{health['avg_predicted_rul']:.0f}", "Avg Predicted RUL"), unsafe_allow_html=True)
    c4.markdown(kpi(f"{eval_meta['rmse']}", "Model RMSE (cycles)"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Fleet Health — Predicted RUL per Engine**")
    st.plotly_chart(create_fleet_grid(fleet_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    for col, status in zip([col_a, col_b, col_c], ["Critical", "Warning", "Healthy"]):
        count = int((fleet_df["status"] == status).sum())
        color = STATUS_COLORS[status]
        col.markdown(kpi(count, f"{status} Engines", color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Maintenance Priority Schedule**")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    schedule = generate_maintenance_schedule(fleet_df)
    st.dataframe(
        schedule[["unit", "predicted_rul", "status", "maintenance_window"]].head(20),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ── Tab 2: Engine Deep Dive ───────────────────────────────────────────────────
with tab2:
    all_units = sorted(fleet_df["unit"].unique())
    selected_unit = st.selectbox("Select Engine", all_units, index=0)

    engine_preds = loader.get_engine_prediction(selected_unit)
    engine_raw   = cmapss["test"][cmapss["test"]["unit"] == selected_unit]
    last_pred    = float(engine_preds["predicted_rul"].iloc[-1])
    rec          = loader.get_maintenance_recommendation(last_pred)
    status       = loader._rul_to_status(last_pred)

    rec_class = {"Critical": "critical", "Warning": "warning", "Healthy": ""}.get(status, "")

    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi(int(engine_raw["cycle"].max()), "Cycles Logged"), unsafe_allow_html=True)
    k2.markdown(kpi(f"{last_pred:.0f}", "Predicted RUL", STATUS_COLORS[status]), unsafe_allow_html=True)
    k3.markdown(kpi(status, "Health Status", STATUS_COLORS[status]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="rec-box {rec_class}">{rec}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_rul, col_sensor = st.columns(2)
    with col_rul:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(create_rul_comparison(engine_preds, selected_unit), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sensor:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(
            create_sensor_dashboard(engine_raw, cmapss["useful_sensors"]),
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ── Tab 3: Model Performance ─────────────────────────────────────────────────
with tab3:
    comparison = compare_to_baseline(eval_meta)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi(eval_meta["rmse"], "LSTM RMSE"), unsafe_allow_html=True)
    c2.markdown(kpi(eval_meta["mae"],  "LSTM MAE"),  unsafe_allow_html=True)
    c3.markdown(kpi(eval_meta["r2"],   "R² Score"),  unsafe_allow_html=True)
    c4.markdown(kpi(f"{comparison['improvement_pct']}%", "vs Naive Baseline"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_curves, col_scatter = st.columns(2)
    with col_curves:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(create_training_curves(history), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_scatter:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(create_prediction_scatter(all_preds), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.plotly_chart(create_error_histogram(all_preds), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("**Model Configuration**")
    col_meta = st.columns(3)
    meta_items = [
        ("Architecture", eval_meta["model"]),
        ("Sequence Length", f"{eval_meta['sequence_length']} cycles"),
        ("Features Used", f"{eval_meta['features_used']} sensors"),
        ("Training Epochs", str(eval_meta["training_epochs"])),
        ("RUL Cap", f"{eval_meta['rul_cap']} cycles"),
        ("Baseline RMSE", str(eval_meta["baseline_rmse"])),
    ]
    for i, (label, val) in enumerate(meta_items):
        col_meta[i % 3].markdown(kpi(val, label), unsafe_allow_html=True)


# ── Tab 4: How It Works ───────────────────────────────────────────────────────
with tab4:
    st.markdown("### How DeepPulse Works")

    with st.expander("What is Remaining Useful Life (RUL)?", expanded=True):
        st.markdown("""
RUL is the number of operational cycles left before an engine needs maintenance. Predicting it accurately means:

- **Avoiding failures** — catch degradation before it becomes a breakdown
- **Avoiding over-maintenance** — don't service an engine that still has 80 cycles left

The NASA C-MAPSS dataset simulates 100 turbofan engines running to failure. Each row is one flight cycle with 21 sensor readings.
""")

    with st.expander("LSTM Architecture"):
        st.markdown("""
**Why LSTM for RUL prediction:**

Sensor readings form a time series — the *sequence* of how a sensor changes over the last 30 cycles matters more than a single snapshot. LSTM (Long Short-Term Memory) networks are designed to learn these sequential patterns.

```
Input: (batch, 30 cycles, 14 sensors)
  → LSTM layer 1: 128 units, return_sequences=True
  → Dropout 0.2
  → LSTM layer 2: 64 units
  → Dropout 0.2
  → Dense 1 (linear output = predicted RUL)
```

**Why cap RUL at 125?** Early in engine life, the actual RUL might be 300+ cycles but the engine shows no degradation signal. Capping at 125 focuses the model on the degradation window where prediction matters.
""")

    with st.expander("Feature Engineering"):
        st.markdown("""
From 21 raw sensors, 7 are dropped because they're flat (constant across all engines):
`s1, s5, s6, s10, s16, s18, s19`

The remaining 14 sensors are **z-score normalized** using training set statistics — this prevents any single sensor from dominating the loss because of its scale.

The input window is the last **30 cycles** per engine. Shorter windows miss the trend; longer windows add noise and slow training.
""")

    with st.expander("NASA Scoring Function"):
        st.markdown("""
NASA uses an asymmetric scoring function that penalizes **late predictions** (predicting more life than actually remains) more than early ones. The standard RMSE we report is simpler and more intuitive.

| Metric | This model | Naive baseline |
|--------|-----------|---------------|
| RMSE   | 12.6 cycles | 40.1 cycles |
| MAE    | 10.1 cycles | — |
| R²     | 0.90 | 0 |
""")

    st.markdown("---")
    st.markdown("**Colab Notebook** - full LSTM training pipeline: `notebooks/lstm_training.ipynb`")
    st.markdown("Built by **Amar Ismail** | [GitHub](https://github.com/amar-ai-engineer) | [LinkedIn](https://www.linkedin.com/in/amar-ai-engineer/)")
