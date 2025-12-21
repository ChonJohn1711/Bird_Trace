from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import try_add_latlon_utm32628

FEATURES = [
    "external-temperature",
    "ground-speed",
    "height-above-msl",
    "gls:light-level",
    "sin_heading",
    "cos_heading",
    "sin_hour",
    "cos_hour",
    "sin_day",
    "cos_day",
    "sin_month",
    "cos_month",
    "distance",
    "time_of_day",
    "season",
]
TARGETS = ["x_m", "y_m"]

FEATURES_STD = ["external-temperature", "gls:light-level", "distance"]
FEATURES_ROBUST = ["ground-speed", "height-above-msl"]
LABEL_COLS = ["time_of_day", "season"]

@dataclass
class InferenceNotes:
    used_model: str
    used_preprocessing: bool
    used_target_scaler: bool
    messages: List[str]

def _encode_labels(df: pd.DataFrame, preprocessing: Optional[Dict[str, Any]], notes: InferenceNotes) -> pd.DataFrame:
    out = df.copy()
    if preprocessing and isinstance(preprocessing, dict) and "label_encoders" in preprocessing:
        le_dict = preprocessing["label_encoders"]
        for col in LABEL_COLS:
            if col in out.columns and col in le_dict:
                out[col] = le_dict[col].transform(out[col].astype(str))
                notes.messages.append(f"Encoded {col} using saved LabelEncoder.")
            else:
                notes.messages.append(f"LabelEncoder for {col} not found; leaving as-is.")
    else:
        fallback_time = {"night":0, "morning":1, "afternoon":2, "evening":3}
        fallback_season = {"spring":0, "summer":1, "fall":2, "winter":3}
        if "time_of_day" in out.columns:
            out["time_of_day"] = out["time_of_day"].astype(str).map(fallback_time).fillna(0).astype(int)
            notes.messages.append("Encoded time_of_day using fallback mapping.")
        if "season" in out.columns:
            out["season"] = out["season"].astype(str).map(fallback_season).fillna(0).astype(int)
            notes.messages.append("Encoded season using fallback mapping.")
    return out

def _apply_scalers(df: pd.DataFrame, preprocessing: Optional[Dict[str, Any]], target_scaler: Optional[Any], notes: InferenceNotes) -> pd.DataFrame:
    out = df.copy()
    if not preprocessing or not isinstance(preprocessing, dict):
        notes.messages.append("No preprocessing scalers found. Skipping scaling.")
        return out

    std = preprocessing.get("std", None)
    robust = preprocessing.get("robust", None)

    try:
        if std is not None:
            out[FEATURES_STD] = std.transform(out[FEATURES_STD].to_numpy())
        if robust is not None:
            out[FEATURES_ROBUST] = robust.transform(out[FEATURES_ROBUST].to_numpy())
        notes.messages.append("Applied saved feature scalers (std + robust).")
        notes.used_preprocessing = True
    except Exception as e:
        notes.messages.append(f"Scaling failed ({e}). Skipping scaling.")

    if target_scaler is not None:
        try:
            out[TARGETS] = target_scaler.transform(out[TARGETS].to_numpy())
            notes.messages.append("Scaled past x_m/y_m using target scaler (for model input).")
        except Exception as e:
            notes.messages.append(f"Target-scaling input failed ({e}). Proceeding without scaling x_m/y_m.")
    return out

def build_model_input(history_window: pd.DataFrame,
                      preprocessing: Optional[Dict[str, Any]],
                      target_scaler: Optional[Any],
                      notes: InferenceNotes) -> np.ndarray:
    for c in FEATURES + TARGETS:
        if c not in history_window.columns:
            raise ValueError(f"Missing required column in history window: {c}")

    df = history_window.copy()
    df = _encode_labels(df, preprocessing, notes)
    df = _apply_scalers(df, preprocessing, target_scaler, notes)

    x_3d = df[FEATURES + TARGETS].to_numpy(dtype=float)  # (48, 17)
    return x_3d.reshape(1, -1)  # (1, 816)

def run_prediction(model: Any,
                   x_flat: np.ndarray,
                   horizon: int,
                   target_scaler: Optional[Any],
                   notes: InferenceNotes) -> np.ndarray:
    y_flat = np.asarray(model.predict(x_flat)).reshape(-1)
    if y_flat.shape[0] != horizon * 2:
        raise ValueError(f"Unexpected model output size: got {y_flat.shape[0]}, expected {horizon*2}")

    y_2d = y_flat.reshape(horizon, 2)

    if target_scaler is not None:
        try:
            y_2d = target_scaler.inverse_transform(y_2d)
            notes.used_target_scaler = True
            notes.messages.append("Inverse-transformed predicted x_m/y_m to original space using target scaler.")
        except Exception as e:
            notes.messages.append(f"Inverse-transform failed ({e}). Returning predicted x_m/y_m in scaled space.")
    else:
        notes.messages.append("No target scaler found. Returning predicted x_m/y_m in scaled space.")
    return y_2d

def assemble_tracks(history_window: pd.DataFrame, pred_xy: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hist = history_window[["timestamp", "x_m", "y_m"]].copy()

    last_ts = pd.to_datetime(history_window["timestamp"].iloc[-1])
    pred_ts = last_ts + pd.to_timedelta(np.arange(1, len(pred_xy)+1), unit="h")

    pred = pd.DataFrame({
        "timestamp": pred_ts,
        "x_m": pred_xy[:,0],
        "y_m": pred_xy[:,1],
    })

    hist = try_add_latlon_utm32628(hist)
    pred = try_add_latlon_utm32628(pred)
    return hist, pred

def summarize_prediction(pred_df: pd.DataFrame) -> Dict[str, float]:
    x = pred_df["x_m"].to_numpy(dtype=float)
    y = pred_df["y_m"].to_numpy(dtype=float)
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx*dx + dy*dy)
    total = float(dist.sum()) if dist.size else 0.0
    avg = float(dist.mean()) if dist.size else 0.0
    return {"pred_total_distance": total, "pred_avg_step_distance": avg}
