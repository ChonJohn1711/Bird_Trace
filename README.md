# Bird Flight Path Prediction Demo (Web)

A web application that predicts a bird’s flight path from historical data and displays the predicted trajectory with an animation (a marker moving along the flight path).

## Objective
- Input: 48 hours of historical data (`x_m`, `y_m`, and influencing factors).
- Output: predicted positions for the next 24 hours.
- UI: display the flight path and animation.

## Map display modes
The frontend chooses the rendering method based on the input data format:
- If `x_m/y_m` looks like Web Mercator (|x|, |y| ≤ 20037508), it converts to lat/lon and displays on OpenStreetMap (internet required).
- If `x_m/y_m` looks like lat/lon (in degrees), it displays directly.
- If neither matches, it falls back to a simple XY plane (CRS.Simple).

Note: The detection and conversion mechanism depends on the frontend/backend code. Please cross-check `static/` and `app.py` to ensure a 100% match.

## Directory structure
- `app.py`: FastAPI backend + serves the static frontend
- `static/`: HTML/CSS/JS (Leaflet)
- `models/`: models and preprocessing artifacts (scalers/encoders)

## Model (pre-trained) — download
- LinearRegression_model_48-24: [here](https://drive.google.com/file/d/18PwyHzJKXKjCZi7bmIdFX3YCRUK7_eG8/view?usp=sharing)
- KNN_model_48-24: [here](https://drive.google.com/file/d/1GoMDA2_zW-Sp1GqqvcrjbWZw6p5PjDep/view?usp=sharing)
- MLP_model_48-24: [here](https://drive.google.com/file/d/145zxd41g_gEJZiEmMWOpZnOseXCQ5qgY/view?usp=sharing)
- RandomForest_model_48-24: [here](https://drive.google.com/file/d/1_rpf9QF_q8ACAAxK-6AUgi3FKyCC-uFF/view?usp=sharing)
- XGBoost_model_48-24: [here](https://drive.google.com/file/d/1Uah9DVLhmp6MxZrLDxYiHpqVSEyYlf8W/view?usp=sharing)

## Using real models (joblib/pickle)

The backend can support loading models from the `models/` directory (please verify `app.py` to confirm the exact priority order and fallback conditions):

- `*.joblib`
- `*.pkl`

If inference fails, the app may fall back to a heuristic predictor so the demo still works (verify in the code to confirm).

## “288 vs 816 features” error (cause)

The pipeline creates input using a 48-hour sliding window and flattens it:

- Each timestep includes `features + target` = 15 + 2 = 17 columns
- `INPUT_WINDOW = 48`

Classic ML model input: 48 × 17 = 816 features

The old demo only sent 6 columns per timestep: 48 × 6 = 288

## Input schema (correct pipeline order)

Each historical row is standardized in the order `df[features + target]`:

1. `external_temperature`
2. `ground_speed`
3. `height_above_msl`
4. `gls_light_level`
5. `sin_heading`
6. `cos_heading`
7. `sin_hour`
8. `cos_hour`
9. `sin_day`
10. `cos_day`
11. `sin_month`
12. `cos_month`
13. `distance`
14. `time_of_day_code` (from `time_of_day`)
15. `season_code` (from `season`)
16. `x_m`
17. `y_m`

The demo takes the last 48 rows, pads if needed, then flattens to shape `(1, 816)` (verify the padding rule in the code).

## Upload CSV

The CSV can use the same header as the dataset. The app may support normalizing column names (e.g., accepting `-` and `:`), but you should verify this in the code.

Minimum recommended fields:

- `timestamp`, `x_m`, `y_m`

If sin/cos columns are missing, the app may generate them from `timestamp` (verify the formulas and periodicity in the code).  
If `distance` is missing, the app may compute it from `x_m/y_m` (verify the computation method in the code).

## Preprocessing (to match training)

The (described) pipeline uses:

- StandardScaler: `external-temperature`, `gls:light-level`, `distance`
- RobustScaler: `ground-speed`, `height-above-msl`
- StandardScaler for targets: `x_m`, `y_m`
- LabelEncoder: `time_of_day`, `season`

To allow the backend to reproduce preprocessing correctly, save artifacts into `models/`:

```python
import joblib

joblib.dump(scalers_encoders, "models/preprocessing.joblib")
joblib.dump(scaler_target, "models/scaler_target.joblib")
````

The backend will automatically look for:

* `models/preprocessing.joblib`
* `models/scaler_target.joblib`

If these two files are missing, the demo may still run the model but skip scaling/inverse-scaling (and show a note in the UI). Verify the behavior in the code.

### Artifacts (already saved) — download

* preprocessing: [here](https://drive.google.com/file/d/1Ea4Vu8Tn_w_buWuPhX0AG9SyCEaf_o8a/view?usp=sharing)
* scaler_target: [here](https://drive.google.com/file/d/1mxfsw5o5RcavgJ_8x98h4EkJX0IkNI6H/view?usp=sharing)

## Run locally

Requirements: Python 3.10+ (recommended)

```bash
cd Bird_Trace
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
uvicorn app:app --reload
```

Open: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## API

* `GET /api/health`: model + preprocessing artifacts status
* `GET /api/sample`: 48h sample data
* `POST /api/predict`: run prediction

Tip: if you use FastAPI, you can check the OpenAPI docs at `/docs` to ensure the README matches the request/response schema.
