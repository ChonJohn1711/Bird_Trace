# Bird Flight Path Prediction Demo (48h -> 24h)

## What this demo does
- Automatically loads a fixed dataset from `data/data_preprocessing_WithoutNormalization.csv`
- You enter a datetime (date/month/year/hour/minute).
- Backend silently floors minutes down to the hour.
- Loads 48 hours of history before the forecast start time and predicts 24 hours ahead.
- Shows map + animated marker + stats table + CSV export.

## Run
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```
Open: http://127.0.0.1:8000

## Data (fixed, auto-load)
Put your dataset file here:
- `data/data_preprocessing_WithoutNormalization.csv`

## Models
Put your trained joblib models in `models/`.

Recommended artifacts (so scaling/encoding matches training):
- `models/preprocessing.joblib`
- `models/scaler_target.joblib`
