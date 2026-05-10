import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

STATUS_COLORS = {"Critical": "#ef4444", "Warning": "#f59e0b", "Healthy": "#22c55e"}


def create_fleet_grid(fleet_df: pd.DataFrame) -> go.Figure:
    """Scatter grid of all engines colored by health status."""
    color_map = fleet_df["status"].map(STATUS_COLORS)
    fig = go.Figure()
    for status, color in STATUS_COLORS.items():
        sub = fleet_df[fleet_df["status"] == status]
        fig.add_trace(go.Scatter(
            x=sub["unit"],
            y=sub["predicted_rul"],
            mode="markers",
            name=status,
            marker=dict(color=color, size=10, line=dict(width=1, color="white")),
            text=[f"Engine {u}<br>Predicted RUL: {r:.0f} cycles" for u, r in zip(sub["unit"], sub["predicted_rul"])],
            hoverinfo="text",
        ))
    fig.add_hline(y=20, line_dash="dash", line_color="#ef4444", annotation_text="Critical <20")
    fig.add_hline(y=50, line_dash="dash", line_color="#f59e0b", annotation_text="Warning <50")
    fig.update_layout(
        height=350,
        margin=dict(t=20, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Engine Unit",
        yaxis_title="Predicted RUL (cycles)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_rul_comparison(engine_df: pd.DataFrame, unit: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=engine_df["cycle"], y=engine_df["actual_rul"],
        mode="lines", name="Actual RUL",
        line=dict(color="#0ea5e9", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=engine_df["cycle"], y=engine_df["predicted_rul"],
        mode="lines", name="LSTM Predicted",
        line=dict(color="#f59e0b", width=2, dash="dot"),
    ))
    fig.add_hline(y=20, line_dash="dash", line_color="#ef4444", annotation_text="Critical threshold")
    fig.update_layout(
        title=f"Engine {unit} — Actual vs Predicted RUL",
        height=320,
        margin=dict(t=40, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Cycle",
        yaxis_title="RUL (cycles)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_sensor_dashboard(engine_raw_df: pd.DataFrame, sensors: list) -> go.Figure:
    """Line chart of normalized sensor readings for one engine."""
    plot_sensors = sensors[:6]
    colors = ["#0ea5e9", "#8b5cf6", "#f59e0b", "#22c55e", "#ef4444", "#64748b"]
    fig = go.Figure()
    for i, s in enumerate(plot_sensors):
        if s in engine_raw_df.columns:
            fig.add_trace(go.Scatter(
                x=engine_raw_df["cycle"], y=engine_raw_df[s],
                mode="lines", name=s.upper(),
                line=dict(color=colors[i % len(colors)], width=1.5),
            ))
    fig.update_layout(
        title="Sensor Trends (normalized)",
        height=300,
        margin=dict(t=40, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Cycle",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_training_curves(history: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["epoch"], y=history["train_loss"],
        mode="lines", name="Train Loss",
        line=dict(color="#0ea5e9", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=history["epoch"], y=history["val_loss"],
        mode="lines", name="Val Loss",
        line=dict(color="#ef4444", width=2, dash="dot"),
    ))
    fig.update_layout(
        title="LSTM Training Loss (MSE)",
        height=300,
        margin=dict(t=40, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_prediction_scatter(all_preds_df: pd.DataFrame) -> go.Figure:
    # Use last-cycle predictions only for the scatter
    last = all_preds_df.sort_values("cycle").groupby("unit").last().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=last["actual_rul"], y=last["predicted_rul"],
        mode="markers",
        marker=dict(color="#0ea5e9", size=8, opacity=0.7),
        name="Engines",
        hovertemplate="Actual: %{x}<br>Predicted: %{y}<extra></extra>",
    ))
    max_val = int(max(last["actual_rul"].max(), last["predicted_rul"].max())) + 10
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", name="Perfect Prediction",
        line=dict(color="#64748b", dash="dash"),
    ))
    fig.update_layout(
        title="Actual vs Predicted RUL (test set, last cycle)",
        height=320,
        margin=dict(t=40, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Actual RUL",
        yaxis_title="Predicted RUL",
    )
    return fig


def create_error_histogram(all_preds_df: pd.DataFrame) -> go.Figure:
    last = all_preds_df.sort_values("cycle").groupby("unit").last().reset_index()
    errors = last["predicted_rul"] - last["actual_rul"]
    fig = go.Figure(go.Histogram(
        x=errors, nbinsx=20,
        marker_color="#8b5cf6", opacity=0.8,
        name="Prediction Error",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#0f172a")
    fig.update_layout(
        title="Prediction Error Distribution (predicted − actual)",
        height=300,
        margin=dict(t=40, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Error (cycles)",
        yaxis_title="Count",
    )
    return fig
