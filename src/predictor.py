import json
import os
import pandas as pd


class PredictionLoader:
    def __init__(
        self,
        predictions_path: str = "data/predictions/lstm_predictions_FD001.csv",
        history_path:     str = "data/predictions/training_history.json",
        eval_path:        str = "data/predictions/model_evaluation.json",
    ):
        self._pred_path = predictions_path
        self._history_path = history_path
        self._eval_path = eval_path
        self._predictions: pd.DataFrame | None = None
        self._history: dict | None = None
        self._eval: dict | None = None

    def _load_predictions(self):
        if self._predictions is None:
            self._predictions = pd.read_csv(self._pred_path)

    def _load_history(self):
        if self._history is None:
            with open(self._history_path) as f:
                self._history = json.load(f)

    def _load_eval(self):
        if self._eval is None:
            with open(self._eval_path) as f:
                self._eval = json.load(f)

    def get_all_predictions(self) -> pd.DataFrame:
        self._load_predictions()
        return self._predictions.copy()

    def get_engine_prediction(self, unit: int) -> pd.DataFrame:
        self._load_predictions()
        return self._predictions[self._predictions["unit"] == unit].copy()

    def get_fleet_status(self) -> pd.DataFrame:
        """Return last-cycle prediction for each engine with health status label."""
        self._load_predictions()
        df = self._predictions.copy()
        last = df.sort_values("cycle").groupby("unit").last().reset_index()
        last["status"] = last["predicted_rul"].apply(self._rul_to_status)
        return last

    def get_training_history(self) -> dict:
        self._load_history()
        return self._history

    def get_evaluation(self) -> dict:
        self._load_eval()
        return self._eval

    @staticmethod
    def _rul_to_status(rul: float) -> str:
        if rul < 20:
            return "Critical"
        if rul < 50:
            return "Warning"
        return "Healthy"

    @staticmethod
    def get_maintenance_recommendation(predicted_rul: float) -> str:
        if predicted_rul < 10:
            return "Immediate shutdown required. Schedule emergency maintenance."
        if predicted_rul < 20:
            return "Schedule maintenance within the next 48 hours."
        if predicted_rul < 50:
            return "Plan maintenance within 2 weeks. Monitor sensor trends closely."
        if predicted_rul < 80:
            return "Routine maintenance check recommended next scheduled window."
        return "Engine is healthy. No immediate action needed."
