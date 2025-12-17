from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import settings
from backend.data_store import DataStore
from backend.model_loader import load_artifacts, load_model
from backend.inference import InferenceNotes, build_model_input, run_prediction, assemble_tracks, summarize_prediction
from backend.utils import floor_to_hour, parse_user_datetime

app = FastAPI(title="Bird Flight Path Prediction Demo")
app.mount("/static", StaticFiles(directory="static"), name="static")

DATA = DataStore(settings.dataset_path)
ART = None

@app.on_event("startup")
def _startup():
    global ART
    try:
        DATA.load()
        DATA.ensure_derived_columns()
    except Exception:
        DATA.info = None
    ART = load_artifacts(settings.models_dir)

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("static/index.html")

@app.get("/api/dataset_info")
def dataset_info():
    if DATA.info is None:
        return {"ok": False, "error": f"Dataset not loaded. Expected at: {settings.dataset_path}"}
    return {
        "ok": True,
        "path": DATA.info.path,
        "rows": DATA.info.rows,
        "start_ts": DATA.info.start_ts,
        "end_ts": DATA.info.end_ts,
        "columns": DATA.info.columns,
        "default_datetime": DATA.get_default_time().strftime("%Y-%m-%dT%H:%M"),
    }

@app.get("/api/models")
def list_models():
    global ART
    if ART is None:
        ART = load_artifacts(settings.models_dir)
    return {
        "models": ART.model_names if ART else [],
        "has_preprocessing": bool(ART and ART.preprocessing is not None),
        "has_target_scaler": bool(ART and ART.target_scaler is not None),
    }

class PredictRequest(BaseModel):
    datetime: str
    model: str | None = None

@app.post("/api/predict")
def predict(req: PredictRequest):
    if DATA.info is None:
        raise HTTPException(status_code=400, detail=f"Dataset not loaded at {settings.dataset_path}")

    try:
        user_dt = parse_user_datetime(req.datetime)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    forecast_start = floor_to_hour(user_dt)

    global ART
    if ART is None:
        ART = load_artifacts(settings.models_dir)

    model_name = req.model or (ART.model_names[0] if ART and ART.model_names else None)
    if not model_name:
        raise HTTPException(status_code=400, detail=f"No models found in {settings.models_dir}")

    try:
        model = load_model(settings.models_dir, model_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    notes = InferenceNotes(
        used_model=model_name,
        used_preprocessing=False,
        used_target_scaler=False,
        messages=["Đã chạy dự đoán.", f"Used_model = {model_name}"],
    )

    try:
        history_window = DATA.get_window_strict(forecast_start, settings.default_input_window)
    except Exception as e:
        # Some datasets have missing hourly records; for demo, fall back to the last N records.
        notes.messages.append(
            f"History window not complete on hourly grid ({e}). Falling back to last {settings.default_input_window} records before forecast_start."
        )
        try:
            history_window = DATA.get_window_last_records(forecast_start, settings.default_input_window)
        except Exception as e2:
            raise HTTPException(status_code=400, detail=str(e2))

    preprocessing = ART.preprocessing if ART else None
    target_scaler = ART.target_scaler if ART else None

    if preprocessing is not None:
        notes.messages.append("Loaded preprocessing artifact: preprocessing.joblib")
    else:
        notes.messages.append("No preprocessing artifact found.")

    if target_scaler is not None:
        notes.messages.append("Loaded target scaler: scaler_target.joblib")
    else:
        notes.messages.append("No target scaler found.")

    try:
        x_flat = build_model_input(history_window, preprocessing, target_scaler, notes)
        pred_xy = run_prediction(model, x_flat, settings.default_horizon, target_scaler, notes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    hist_df, pred_df = assemble_tracks(history_window, pred_xy)
    summary = summarize_prediction(pred_df)

    return {
        "ok": True,
        "input_datetime": req.datetime,  # do not mention rounding
        "horizon_hours": settings.default_horizon,
        "history_hours": settings.default_input_window,
        "history": hist_df.to_dict(orient="records"),
        "prediction": pred_df.to_dict(orient="records"),
        "summary": summary,
        "notes": {
            "used_model": notes.used_model,
            "used_preprocessing": notes.used_preprocessing,
            "used_target_scaler": notes.used_target_scaler,
            "messages": notes.messages,
        },
    }

@app.get("/api/prediction.csv")
def prediction_csv(datetime: str, model: str | None = None):
    req = PredictRequest(datetime=datetime, model=model)
    res = predict(req)
    import pandas as pd
    df = pd.DataFrame(res["prediction"])
    tmp_path = "static/_prediction_export.csv"
    df.to_csv(tmp_path, index=False)
    return FileResponse(tmp_path, filename="prediction.csv", media_type="text/csv")


@app.get("/api/prediction.xlsx")
def prediction_xlsx(datetime: str, model: str | None = None):
    """Export prediction to an Excel file (xlsx)."""
    req = PredictRequest(datetime=datetime, model=model)
    res = predict(req)
    import pandas as pd
    df = pd.DataFrame(res["prediction"])
    tmp_path = "static/_prediction_export.xlsx"
    # Uses openpyxl engine (declared in requirements)
    with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="prediction")
    return FileResponse(
        tmp_path,
        filename="prediction.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
