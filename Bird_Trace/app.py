
import os
import math
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

try:
    from pydantic import BaseModel, Field
    # Pydantic v2 compatibility: allow field names like model_name
    from pydantic import ConfigDict
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore
    Field = lambda *a, **k: None  # type: ignore
    ConfigDict = None  # type: ignore

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_INPUT_WINDOW = 48  # 48h history

# ----------------------------
# API Schemas
# ----------------------------

class Record(BaseModel):
    """One historical timestep."""

    model_config = ConfigDict(extra="allow", protected_namespaces=()) if ConfigDict else None

    timestamp: Optional[str] = Field(default=None, description="ISO string or any parseable datetime")
    x_m: float
    y_m: float

    # a few common optional fields (others can come via extra)
    external_temperature: Optional[float] = None
    ground_speed: Optional[float] = None
    height_above_msl: Optional[float] = None
    gls_light_level: Optional[float] = None

    sin_heading: Optional[float] = None
    cos_heading: Optional[float] = None

    sin_hour: Optional[float] = None
    cos_hour: Optional[float] = None
    sin_day: Optional[float] = None
    cos_day: Optional[float] = None
    sin_month: Optional[float] = None
    cos_month: Optional[float] = None

    distance: Optional[float] = None

    # can be string ("night"/"fall") or already-encoded numeric
    time_of_day: Optional[Any] = None
    season: Optional[Any] = None


class PredictRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=()) if ConfigDict else None

    records: List[Record] = Field(..., description="Historical records (ideally last 48h)")
    horizon_hours: int = Field(default=24, ge=1, le=168, description="How many hours to predict")
    model_name: Optional[str] = Field(default=None, description="Specific model file in /models")


class PointOut(BaseModel):
    timestamp: str
    x_m: float
    y_m: float


class PredictResponse(BaseModel):
    used_model: str
    predicted: List[PointOut]
    notes: List[str] = Field(default_factory=list)


# ----------------------------
# Model + preprocessing artifacts loading
# ----------------------------

def _list_model_files() -> List[str]:
    if not os.path.isdir(MODELS_DIR):
        return []
    out: List[str] = []
    for fn in os.listdir(MODELS_DIR):
        if fn.startswith("."):
            continue
        p = os.path.join(MODELS_DIR, fn)
        if os.path.isfile(p):
            out.append(fn)
    return sorted(out)


def _load_model(file_name: Optional[str]) -> Tuple[Optional[Any], str, List[str]]:
    notes: List[str] = []
    files = _list_model_files()
    if not files:
        return None, "heuristic", ["No files found in /models. Using heuristic predictor."]

    # prefer user-chosen, else first supported format
    chosen = file_name or next((f for f in files if f.lower().endswith((".joblib", ".pkl"))), None)
    if chosen is None:
        return None, "heuristic", [
            "Found model files, but none with supported extensions (.joblib, .pkl). Using heuristic predictor.",
            f"Files in /models: {', '.join(files)}",
        ]

    path = os.path.join(MODELS_DIR, chosen)
    ext = os.path.splitext(chosen)[1].lower()

    try:
        if ext == ".joblib":
            if joblib is None:
                return None, "heuristic", ["joblib is not available. Using heuristic predictor."]
            model = joblib.load(path)
            notes.append(f"Loaded joblib model: {chosen}")
            return model, chosen, notes
        if ext == ".pkl":
            with open(path, "rb") as f:
                model = pickle.load(f)
            notes.append(f"Loaded pickle model: {chosen}")
            return model, chosen, notes
    except Exception as e:
        return None, "heuristic", [f"Failed to load model '{chosen}': {e}. Using heuristic predictor."]

    return None, "heuristic", [f"Unsupported model extension '{ext}'. Using heuristic predictor."]


def _try_load_joblib_any(names: List[str]) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    """Return (obj, filename, error)"""
    if joblib is None:
        return None, None, "joblib is not available"
    for n in names:
        p = os.path.join(MODELS_DIR, n)
        if os.path.isfile(p):
            try:
                return joblib.load(p), n, None
            except Exception as e:
                return None, n, str(e)
    return None, None, None


# Training-time artifacts (optional):
# - scalers_encoders: dict with keys std, robust, target, label_encoders
# - scaler_target: StandardScaler for inverse_transform (targets)
PREP: Dict[str, Any] = {}
PREP_NOTES: List[str] = []


def _load_preprocessing_artifacts() -> None:
    global PREP, PREP_NOTES
    PREP = {}
    PREP_NOTES = []

    scalers, fn, err = _try_load_joblib_any([
        "scalers_encoders.joblib",
        "scalers_encoders.pkl",
        "preprocessing.joblib",
        "preprocessors.joblib",
    ])
    if scalers is not None:
        PREP["scalers_encoders"] = scalers
        PREP_NOTES.append(f"Loaded preprocessing artifact: {fn}")
    elif fn and err:
        PREP_NOTES.append(f"Found {fn} but failed to load: {err}. Will skip preprocessing.")

    st, fn2, err2 = _try_load_joblib_any([
        "scaler_target.joblib",
        "target_scaler.joblib",
    ])
    if st is not None:
        PREP["scaler_target"] = st
        PREP_NOTES.append(f"Loaded target scaler: {fn2}")
    elif fn2 and err2:
        PREP_NOTES.append(f"Found {fn2} but failed to load: {err2}. Will skip inverse_transform.")


_load_preprocessing_artifacts()


# ----------------------------
# Feature handling (match your training pipeline)
# ----------------------------

# Your training order per timestep: features + target
FEATURES_15_ORDER = [
    "external_temperature",
    "ground_speed",
    "height_above_msl",
    "gls_light_level",
    "sin_heading",
    "cos_heading",
    "sin_hour",
    "cos_hour",
    "sin_day",
    "cos_day",
    "sin_month",
    "cos_month",
    "distance",
    "time_of_day_code",
    "season_code",
]
TARGET_2_ORDER = ["x_m", "y_m"]
FEATURES_PLUS_TARGET_17_ORDER = FEATURES_15_ORDER + TARGET_2_ORDER

# A smaller fallback for models trained with a subset
DEFAULT_FEATURE_ORDER_6 = [
    "external_temperature",
    "ground_speed",
    "sin_heading",
    "cos_heading",
    "x_m",
    "y_m",
]

# Common column name normalization
NORMALIZE_KEYS = {
    "external-temperature": "external_temperature",
    "ground-speed": "ground_speed",
    "height-above-msl": "height_above_msl",
    "gls:light-level": "gls_light_level",
}

# Deterministic fallback encodings (only used if LabelEncoders are not available)
TIME_OF_DAY_FALLBACK = {"night": 0.0, "morning": 1.0, "afternoon": 2.0, "evening": 3.0}
SEASON_FALLBACK = {"winter": 0.0, "spring": 1.0, "summer": 2.0, "fall": 3.0, "autumn": 3.0}


def _normalize_record_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kk = k
        if kk in NORMALIZE_KEYS:
            kk = NORMALIZE_KEYS[kk]
        # general normalization
        kk = kk.replace(":", "_").replace("-", "_")
        out[kk] = v
    return out


def _to_dataframe(records: List[Record]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in records:
        dd = r.model_dump()
        dd = _normalize_record_dict(dd)
        rows.append(dd)

    df = pd.DataFrame(rows)

    # Timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def _ensure_time_cyclic_features(df: pd.DataFrame, notes: List[str]) -> None:
    if "timestamp" not in df.columns or df["timestamp"].notna().sum() == 0:
        notes.append("Missing/invalid timestamp. Cannot derive sin/cos hour/day/month.")
        return

    ts = df["timestamp"]

    if "sin_hour" not in df.columns or "cos_hour" not in df.columns:
        hour = ts.dt.hour.fillna(0).astype(int)
        ang = 2 * math.pi * (hour / 24.0)
        df["sin_hour"] = np.sin(ang)
        df["cos_hour"] = np.cos(ang)
        notes.append("Derived sin_hour/cos_hour from timestamp.")

    if "sin_month" not in df.columns or "cos_month" not in df.columns:
        month = ts.dt.month.fillna(1).astype(int)
        ang = 2 * math.pi * (month / 12.0)
        df["sin_month"] = np.sin(ang)
        df["cos_month"] = np.cos(ang)
        notes.append("Derived sin_month/cos_month from timestamp.")

    # Your sample matches 2*pi*day/30.0 (day is day-of-month)
    if "sin_day" not in df.columns or "cos_day" not in df.columns:
        day = ts.dt.day.fillna(1).astype(int)
        ang = 2 * math.pi * (day / 30.0)
        df["sin_day"] = np.sin(ang)
        df["cos_day"] = np.cos(ang)
        notes.append("Derived sin_day/cos_day from timestamp (2*pi*day/30).")


def _ensure_distance(df: pd.DataFrame, notes: List[str]) -> None:
    if "distance" in df.columns and df["distance"].notna().any():
        return
    if "x_m" not in df.columns or "y_m" not in df.columns:
        return

    x = pd.to_numeric(df["x_m"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(df["y_m"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dist = np.zeros(len(df), dtype=float)
    if len(df) >= 2:
        dx = np.diff(x)
        dy = np.diff(y)
        dist[1:] = np.hypot(dx, dy)
    df["distance"] = dist
    notes.append("Derived distance from consecutive (x_m, y_m).")


def _encode_labels(df: pd.DataFrame, notes: List[str]) -> None:
    # Prefer training-time LabelEncoders if available
    label_encoders = None
    se = PREP.get("scalers_encoders")
    if isinstance(se, dict):
        label_encoders = se.get("label_encoders")

    def encode_col(col: str, fallback: Dict[str, float], out_col: str) -> None:
        if out_col in df.columns and df[out_col].notna().any():
            return
        if col not in df.columns:
            df[out_col] = 0.0
            notes.append(f"Missing {col}. Filled {out_col}=0.")
            return

        s = df[col]
        # numeric already
        if pd.api.types.is_numeric_dtype(s):
            df[out_col] = pd.to_numeric(s, errors="coerce").fillna(0.0)
            return

        # try LabelEncoder
        if label_encoders and col in label_encoders:
            le = label_encoders[col]
            try:
                df[out_col] = le.transform(s.astype(str).fillna(""))
                notes.append(f"Encoded {col} using saved LabelEncoder.")
                return
            except Exception as e:
                notes.append(f"LabelEncoder for {col} failed ({e}). Falling back to deterministic map.")

        # fallback map
        df[out_col] = s.astype(str).str.lower().map(fallback).fillna(0.0)
        notes.append(f"Encoded {col} using fallback mapping.")

    encode_col("time_of_day", TIME_OF_DAY_FALLBACK, "time_of_day_code")
    encode_col("season", SEASON_FALLBACK, "season_code")


def _apply_feature_scaling(df: pd.DataFrame, notes: List[str]) -> pd.DataFrame:
    """Apply the same scaling you used in training, if artifacts are available."""
    se = PREP.get("scalers_encoders")
    if not isinstance(se, dict):
        notes.append("No preprocessing scalers found. Skipping scaling.")
        return df

    df2 = df.copy()
    features_std = ["external_temperature", "gls_light_level", "distance"]
    features_robust = ["ground_speed", "height_above_msl"]
    target_cols = ["x_m", "y_m"]

    # fill missing columns before transform
    for c in features_std + features_robust + target_cols:
        if c not in df2.columns:
            df2[c] = 0.0

    try:
        std = se.get("std")
        robust = se.get("robust")
        target = se.get("target")

        # NOTE:
        # Many sklearn transformers store feature_names_in_. If we pass a DataFrame
        # with different column names (e.g., our normalized external_temperature vs
        # training-time external-temperature), sklearn can raise a feature-name
        # mismatch error.
        #
        # To keep the demo robust across naming conventions, we feed numpy arrays
        # (same column order, same feature count) to bypass strict name checks.
        if std is not None:
            x_std = df2[features_std].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                df2[features_std] = std.transform(x_std)
        if robust is not None:
            x_rob = df2[features_robust].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                df2[features_robust] = robust.transform(x_rob)
        if target is not None:
            x_tgt = df2[target_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                df2[target_cols] = target.transform(x_tgt)

        notes.append("Applied saved scalers to input (std/robust/target).")
        return df2
    except Exception as e:
        notes.append(f"Scaling failed ({e}). Skipping scaling.")
        return df


def _select_feature_order_for_model(model: Any, notes: List[str]) -> List[str]:
    nfi = getattr(model, "n_features_in_", None)
    if isinstance(nfi, (int, np.integer)):
        nfi_int = int(nfi)
        # Most common: 48 timesteps
        if nfi_int % DEFAULT_INPUT_WINDOW == 0:
            per_step = nfi_int // DEFAULT_INPUT_WINDOW
            if per_step == 17:
                return FEATURES_PLUS_TARGET_17_ORDER
            if per_step == 6:
                notes.append("Model expects 6 features/timestep; using small feature order.")
                return DEFAULT_FEATURE_ORDER_6
            # best-effort: truncate known list
            if per_step < len(FEATURES_PLUS_TARGET_17_ORDER):
                notes.append(f"Model expects {per_step} features/timestep; truncating canonical order.")
                return FEATURES_PLUS_TARGET_17_ORDER[:per_step]

        # If it's a flat feature vector not divisible by 48, try 17*48 exactly
        if nfi_int == 816:
            return FEATURES_PLUS_TARGET_17_ORDER

        notes.append(f"Model n_features_in_={nfi_int}. Using canonical 17-feature order as best effort.")

    return FEATURES_PLUS_TARGET_17_ORDER


def _build_window_X(df_hist: pd.DataFrame, feature_cols: List[str], notes: List[str]) -> np.ndarray:
    df = df_hist.copy()

    # Keep last 48h
    if len(df) < DEFAULT_INPUT_WINDOW:
        # pad by repeating first row
        pad_n = DEFAULT_INPUT_WINDOW - len(df)
        pad = pd.concat([df.iloc[[0]]] * pad_n, ignore_index=True)
        df = pd.concat([pad, df], ignore_index=True)
        notes.append(f"History shorter than 48 rows. Padded {pad_n} rows by repeating the earliest record.")
    else:
        df = df.tail(DEFAULT_INPUT_WINDOW).reset_index(drop=True)

    # Ensure all needed columns exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return X.reshape(1, -1)


def _inverse_targets_if_possible(y_scaled: np.ndarray, notes: List[str]) -> np.ndarray:
    st = PREP.get("scaler_target")
    if st is None:
        # try from scalers_encoders
        se = PREP.get("scalers_encoders")
        if isinstance(se, dict):
            st = se.get("target")

    if st is None:
        notes.append("No target scaler found. Returning predicted x_m/y_m in scaled space.")
        return y_scaled

    try:
        return st.inverse_transform(y_scaled)
    except Exception as e:
        notes.append(f"Inverse transform failed ({e}). Returning predicted x_m/y_m in scaled space.")
        return y_scaled


def _model_predict(model: Any, df_hist_raw: pd.DataFrame, horizon_hours: int) -> Tuple[np.ndarray, List[str]]:
    notes: List[str] = []

    # Prepare dataframe
    df = df_hist_raw.copy()

    # derive missing columns
    _ensure_time_cyclic_features(df, notes)
    _ensure_distance(df, notes)
    _encode_labels(df, notes)

    # apply training-time scaling if available
    df_scaled = _apply_feature_scaling(df, notes)

    feat_order = _select_feature_order_for_model(model, notes)
    X = _build_window_X(df_scaled, feat_order, notes)

    if not hasattr(model, "predict"):
        raise ValueError("Loaded model does not have a .predict method")

    y = model.predict(X)
    arr = np.asarray(y)
    arr = np.squeeze(arr)

    # Expect horizon*2 (flattened)
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            raise ValueError(f"Model output length {arr.size} is not even; cannot split into (x,y) pairs")
        pts_scaled = arr.reshape(-1, 2)
    elif arr.ndim == 2:
        if arr.shape[1] == 2:
            pts_scaled = arr
        elif arr.shape[0] == 1 and arr.shape[1] % 2 == 0:
            pts_scaled = arr.reshape(-1, 2)
        else:
            raise ValueError(f"Unexpected model output shape: {arr.shape}")
    else:
        raise ValueError(f"Unexpected model output ndim: {arr.ndim}")

    # Align horizon
    if pts_scaled.shape[0] != horizon_hours:
        notes.append(f"Model returned {pts_scaled.shape[0]} steps; requested {horizon_hours}. Will align by truncating/padding.")
        if pts_scaled.shape[0] > horizon_hours:
            pts_scaled = pts_scaled[:horizon_hours, :]
        else:
            last = pts_scaled[-1:, :]
            pad = np.repeat(last, horizon_hours - pts_scaled.shape[0], axis=0)
            pts_scaled = np.vstack([pts_scaled, pad])

    # inverse to original coordinate space if possible
    pts_actual = _inverse_targets_if_possible(pts_scaled, notes)

    return pts_actual.astype(float), notes


# ----------------------------
# Heuristic fallback
# ----------------------------

def _heuristic_predict(df_hist: pd.DataFrame, horizon_hours: int) -> Tuple[np.ndarray, List[str]]:
    notes: List[str] = []

    if len(df_hist) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 records to predict.")

    df = df_hist.sort_values("timestamp") if "timestamp" in df_hist.columns and df_hist["timestamp"].notna().any() else df_hist

    last = df.iloc[-1]
    x, y = float(last["x_m"]), float(last["y_m"])

    # step size
    dt_hours = 1.0
    if "timestamp" in df.columns and df["timestamp"].notna().sum() >= 2:
        t1 = df.iloc[-2]["timestamp"]
        t2 = df.iloc[-1]["timestamp"]
        if pd.notna(t1) and pd.notna(t2):
            dh = (t2 - t1).total_seconds() / 3600.0
            if 0.1 <= dh <= 6:
                dt_hours = float(dh)
    notes.append(f"Heuristic step size: {dt_hours:.2f}h per prediction step.")

    # speed
    v = last.get("ground_speed", np.nan)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        prev = df.iloc[-2]
        dx = float(last["x_m"]) - float(prev["x_m"])
        dy = float(last["y_m"]) - float(prev["y_m"])
        dist = math.hypot(dx, dy)
        v = dist / max(dt_hours, 1e-6) / 3600.0
        notes.append("No ground_speed found. Estimated speed from last displacement.")
    else:
        v = float(v)

    # heading
    sh = last.get("sin_heading", np.nan)
    ch = last.get("cos_heading", np.nan)
    if sh is not None and ch is not None and not (isinstance(sh, float) and math.isnan(sh)) and not (isinstance(ch, float) and math.isnan(ch)):
        heading = math.atan2(float(sh), float(ch))
    else:
        prev = df.iloc[-2]
        dx = float(last["x_m"]) - float(prev["x_m"])
        dy = float(last["y_m"]) - float(prev["y_m"])
        heading = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-9 else 0.0
        notes.append("Estimated heading from last displacement.")

    points = []
    for i in range(1, horizon_hours + 1):
        meters_per_step = v * 3600.0 * dt_hours
        curve = 0.02 * math.sin(i / 6.0)
        heading_i = heading + curve
        x += meters_per_step * math.cos(heading_i)
        y += meters_per_step * math.sin(heading_i)
        points.append((x, y))

    return np.array(points, dtype=float), notes


# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(title="Bird Flight Path Prediction Demo", version="1.1.1")
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health():
    return {"status": "ok", "models": _list_model_files(), "preprocessing": PREP_NOTES}


@app.get("/api/models")
def list_models():
    return {"models": _list_model_files()}


@app.get("/api/sample")
def sample():
    """Return a deterministic sample with 48 hourly points and full feature columns."""

    base_x, base_y = 10000.0, 8000.0
    records = []

    ts0 = pd.Timestamp("2023-11-18 22:00:00")
    x, y = base_x, base_y
    v = 12.0
    heading = 0.4

    for i in range(DEFAULT_INPUT_WINDOW):
        t = ts0 + pd.Timedelta(hours=i)

        curve = 0.02 * math.sin(i / 5.0)
        heading_i = heading + curve

        dx = v * 3600.0 * math.cos(heading_i)
        dy = v * 3600.0 * math.sin(heading_i)
        x2, y2 = x + dx, y + dy

        temp = 20.0 + 6.0 * math.sin(i / 8.0)
        light = max(0.0, 120.0 * math.sin((t.hour / 24.0) * 2 * math.pi))
        height = 50.0 + 10.0 * math.sin(i / 7.0)

        rec = {
            "timestamp": str(t),
            "external-temperature": float(temp),
            "ground-speed": float(v),
            "height-above-msl": float(height),
            "gls:light-level": float(light),
            "sin_heading": float(math.sin(heading_i)),
            "cos_heading": float(math.cos(heading_i)),
            "x_m": float(x2),
            "y_m": float(y2),
            "time_of_day": "night" if t.hour < 6 else "morning" if t.hour < 12 else "afternoon" if t.hour < 18 else "evening",
            "season": "fall",
        }

        records.append(rec)
        x, y = x2, y2

    # Add distance column for completeness (raw)
    df = pd.DataFrame([_normalize_record_dict(r) for r in records])
    _ensure_distance(df, [])
    for i, d in enumerate(df["distance"].to_numpy()):
        records[i]["distance"] = float(d)

    return {"records": records}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.records) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 historical records.")

    df = _to_dataframe(req.records)

    # Sort + last timestamp
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df = df.sort_values("timestamp")
        last_ts = df["timestamp"].dropna().iloc[-1]
    else:
        df = df.reset_index(drop=True)
        last_ts = pd.Timestamp.utcnow().floor("H")

    model, used_model, notes = _load_model(req.model_name)
    notes = PREP_NOTES + notes

    if model is None:
        pts, extra_notes = _heuristic_predict(df, req.horizon_hours)
        notes.extend(extra_notes)
        used_model = "heuristic"
    else:
        try:
            pts, model_notes = _model_predict(model, df, req.horizon_hours)
            notes.extend(model_notes)
        except Exception as e:
            notes.append(f"Model inference failed ({e}). Falling back to heuristic predictor.")
            pts, extra_notes = _heuristic_predict(df, req.horizon_hours)
            notes.extend(extra_notes)
            used_model = "heuristic"

    # Determine step size for timestamps
    step_hours = 1.0
    if "timestamp" in df.columns and df["timestamp"].notna().sum() >= 2:
        t1 = df.iloc[-2]["timestamp"]
        t2 = df.iloc[-1]["timestamp"]
        if pd.notna(t1) and pd.notna(t2):
            dh = (t2 - t1).total_seconds() / 3600.0
            if 0.1 <= dh <= 6:
                step_hours = float(dh)

    predicted: List[PointOut] = []
    for i in range(req.horizon_hours):
        ts = last_ts + pd.Timedelta(hours=step_hours * (i + 1))
        predicted.append(PointOut(timestamp=str(ts), x_m=float(pts[i, 0]), y_m=float(pts[i, 1])))

    return PredictResponse(used_model=str(used_model), predicted=predicted, notes=notes)
