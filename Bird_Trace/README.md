````md
# Bird Flight Path Prediction Demo (Web)

Demo web dự đoán đường bay chim từ dữ liệu lịch sử, kèm animation marker chạy theo quỹ đạo dự đoán.

## Mục tiêu
- **Input:** 48 giờ lịch sử (`x_m`, `y_m` + các yếu tố ảnh hưởng)
- **Output:** dự đoán vị trí **24 giờ** tiếp theo
- **UI:** hiển thị đường bay + animation

## Chế độ hiển thị bản đồ
Frontend tự chọn cách render theo dữ liệu:
- Nếu `x_m/y_m` giống **Web Mercator** (|x|,|y| ≤ 20037508) → chuyển sang `lat/lon` và hiển thị trên **OpenStreetMap** (cần internet).
- Nếu `x_m/y_m` trông giống **lat/lon** (độ) → hiển thị trực tiếp.
- Nếu không khớp → fallback sang mặt phẳng XY (**CRS.Simple**).

## Cấu trúc thư mục
- `app.py`: FastAPI backend + serve static frontend
- `static/`: HTML/CSS/JS (Leaflet)
- `models/`: model + preprocessing artifacts (scaler/encoder)

## Model (đã train) — tải về
- LinearRegression_model_48-24: https://drive.google.com/file/d/18PwyHzJKXKjCZi7bmIdFX3YCRUK7_eG8/view?usp=sharing
- KNN_model_48-24: https://drive.google.com/file/d/1GoMDA2_zW-Sp1GqqvcrjbWZw6p5PjDep/view?usp=sharing
- MLP_model_48-24: https://drive.google.com/file/d/145zxd41g_gEJZiEmMWOpZnOseXCQ5qgY/view?usp=sharing
- RandomForest_model_48-24: https://drive.google.com/file/d/1_rpf9QF_q8ACAAxK-6AUgi3FKyCC-uFF/view?usp=sharing
- XGBoost_model_48-24: https://drive.google.com/file/d/1Uah9DVLhmp6MxZrLDxYiHpqVSEyYlf8W/view?usp=sharing

## Chạy local
Yêu cầu: **Python 3.10+** (khuyến nghị)

```bash
cd birdflight_demo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
uvicorn app:app --reload
````

Mở: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Dùng model thật (joblib/pickle)

Backend sẽ thử load model trong `models/` theo thứ tự ưu tiên:

1. `*.joblib`
2. `*.pkl`

Nếu inference lỗi, backend sẽ fallback sang **heuristic predictor** để demo vẫn hoạt động.

## Lỗi “288 vs 816 features” (nguyên nhân)

Pipeline tạo input cửa sổ 48 giờ rồi **flatten**:

* Mỗi timestep có `features + target` = 15 + 2 = **17** cột
* `INPUT_WINDOW = 48`

⇒ input cho model dạng ML cổ điển: **48 × 17 = 816 features**

Bản demo cũ chỉ gửi 6 cột/timestep ⇒ **48 × 6 = 288**.

## Schema input (đúng thứ tự pipeline)

Mỗi dòng lịch sử được chuẩn hóa theo đúng `df[features + target]`:

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
14. `time_of_day_code` (từ `time_of_day`)
15. `season_code` (từ `season`)
16. `x_m`
17. `y_m`

Demo lấy **48 dòng cuối**, padding nếu thiếu, rồi flatten thành `(1, 816)`.

## Upload CSV

CSV có thể dùng header giống dataset (chấp nhận cả `-` và `:`), ví dụ:

* `external-temperature`, `ground-speed`, `height-above-msl`, `gls:light-level`

Khuyến nghị tối thiểu có:

* `timestamp`, `x_m`, `y_m`

Nếu thiếu các cột sin/cos, demo sẽ tự sinh từ `timestamp`:

* `sin_hour/cos_hour`: chu kỳ 24
* `sin_day/cos_day`: chu kỳ 30 (theo mẫu bạn đưa)
* `sin_month/cos_month`: chu kỳ 12

Nếu thiếu `distance`, demo tự tính từ `x_m/y_m`.

## Preprocessing (để khớp lúc train)

Pipeline bạn gửi dùng:

* StandardScaler: `external-temperature`, `gls:light-level`, `distance`
* RobustScaler: `ground-speed`, `height-above-msl`
* StandardScaler cho target: `x_m`, `y_m`
* LabelEncoder: `time_of_day`, `season`

Để backend tái hiện đúng preprocess, hãy lưu artifacts vào `models/`:

```python
import joblib

joblib.dump(scalers_encoders, "models/preprocessing.joblib")
joblib.dump(scaler_target, "models/scaler_target.joblib")
```

Backend sẽ tự tìm:

* `models/preprocessing.joblib`
* `models/scaler_target.joblib`

### Artifacts (đã lưu) — tải về

* preprocessing: [https://drive.google.com/file/d/1Ea4Vu8Tn_w_buWuPhX0AG9SyCEaf_o8a/view?usp=sharing](here)
* scaler_target: [https://drive.google.com/file/d/1mxfsw5o5RcavgJ_8x98h4EkJX0IkNI6H/view?usp=sharing](here)

Nếu thiếu 2 file này, demo vẫn chạy model nhưng **bỏ qua scaling/inverse** (và hiển thị ghi chú trong UI).

## API

* `GET /api/health`: trạng thái model + preprocessing artifacts
* `GET /api/sample`: dữ liệu mẫu 48h
* `POST /api/predict`: chạy dự đoán

```
```