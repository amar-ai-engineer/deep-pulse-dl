import pandas as pd
import numpy as np


def compute_fleet_health(fleet_df: pd.DataFrame) -> dict:
    """Summarize fleet health from fleet_status DataFrame."""
    total = len(fleet_df)
    critical = int((fleet_df["status"] == "Critical").sum())
    warning  = int((fleet_df["status"] == "Warning").sum())
    healthy  = int((fleet_df["status"] == "Healthy").sum())
    avg_rul  = float(fleet_df["predicted_rul"].mean())
    return {
        "total_engines": total,
        "critical": critical,
        "warning": warning,
        "healthy": healthy,
        "avg_predicted_rul": round(avg_rul, 1),
        "engines_at_risk": critical + warning,
    }


def compare_to_baseline(eval_meta: dict) -> dict:
    """Return comparison metrics vs naive mean-predict baseline."""
    return {
        "lstm_rmse": eval_meta["rmse"],
        "baseline_rmse": eval_meta["baseline_rmse"],
        "improvement_pct": eval_meta["improvement_vs_baseline_pct"],
        "lstm_r2": eval_meta["r2"],
    }


def generate_maintenance_schedule(fleet_df: pd.DataFrame) -> pd.DataFrame:
    """Build a prioritized maintenance schedule from fleet status."""
    df = fleet_df[["unit", "predicted_rul", "status"]].copy()
    df = df.sort_values("predicted_rul")
    priority_map = {"Critical": 1, "Warning": 2, "Healthy": 3}
    df["priority"] = df["status"].map(priority_map)

    def window(rul):
        if rul < 10:  return "Immediate"
        if rul < 20:  return "Within 48 hours"
        if rul < 50:  return "Within 2 weeks"
        if rul < 80:  return "Next scheduled window"
        return "Routine check only"

    df["maintenance_window"] = df["predicted_rul"].apply(window)
    return df.reset_index(drop=True)
