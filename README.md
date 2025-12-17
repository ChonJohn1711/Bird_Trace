# Bird Flight Path Prediction Demo (Web)

Ứng dụng web dự đoán đường bay chim từ dữ liệu lịch sử và hiển thị quỹ đạo dự đoán kèm animation (marker chạy theo đường bay).

## Mục tiêu
- Input: 48 giờ dữ liệu lịch sử (`x_m`, `y_m` và các yếu tố ảnh hưởng).
- Output: dự đoán vị trí 24 giờ tiếp theo.
- UI: hiển thị đường bay và animation.

## Chế độ hiển thị bản đồ
Frontend lựa chọn cách render theo dạng dữ liệu đầu vào:
- Nếu `x_m/y_m` giống Web Mercator (|x|, |y| ≤ 20037508) thì chuyển sang lat/lon và hiển thị trên OpenStreetMap (cần internet).
- Nếu `x_m/y_m` giống lat/lon (đơn vị độ) thì hiển thị trực tiếp.
- Nếu không khớp, fallback sang mặt phẳng XY (CRS.Simple).

Lưu ý: Cơ chế nhận biết và chuyển đổi phụ thuộc vào code frontend/backend. Hãy đối chiếu với `static/` và `app.py` để đảm bảo khớp 100%.

## Cấu trúc thư mục
- `app.py`: FastAPI backend + serve static frontend
- `static/`: HTML/CSS/JS (Leaflet)
- `models/`: model và preprocessing artifacts (scaler/encoder)

## Model (đã train) — tải về
- LinearRegression_model_48-24: [here](https://drive.google.com/file/d/18PwyHzJKXKjCZi7bmIdFX3YCRUK7_eG8/view?usp=sharing)
- KNN_model_48-24: [here](https://drive.google.com/file/d/1GoMDA2_zW-Sp1GqqvcrjbWZw6p5PjDep/view?usp=sharing)
- MLP_model_48-24: [here](https://drive.google.com/file/d/145zxd41g_gEJZiEmMWOpZnOseXCQ5qgY/view?usp=sharing)
- RandomForest_model_48-24: [here](https://drive.google.com/file/d/1_rpf9QF_q8ACAAxK-6AUgi3FKyCC-uFF/view?usp=sharing)
- XGBoost_model_48-24: [here](https://drive.google.com/file/d/1Uah9DVLhmp6MxZrLDxYiHpqVSEyYlf8W/view?usp=sharing)

## Schema input (đúng thứ tự pipeline)

Mỗi dòng lịch sử được chuẩn hóa theo thứ tự `df[features + target]`:

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

Demo lấy 48 dòng cuối, padding nếu thiếu, rồi flatten thành shape `(1, 816)` (cần đối chiếu code để xác minh padding rule).

## Upload CSV

CSV có thể dùng header giống dataset. Ứng dụng có thể hỗ trợ normalize tên cột (ví dụ chấp nhận `-` và `:`), nhưng cần đối chiếu code để xác minh.

Khuyến nghị tối thiểu:

* `timestamp`, `x_m`, `y_m`

Nếu thiếu các cột sin/cos, ứng dụng có thể tự sinh từ `timestamp` (cần đối chiếu code để xác minh công thức và chu kỳ).
Nếu thiếu `distance`, ứng dụng có thể tự tính từ `x_m/y_m` (cần đối chiếu code để xác minh cách tính).

## Preprocessing (để khớp lúc train)

Pipeline (theo mô tả) sử dụng:

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

Nếu thiếu 2 file này, demo có thể vẫn chạy model nhưng bỏ qua scaling/inverse (và hiển thị ghi chú trong UI). Cần đối chiếu code để xác minh hành vi.

### Artifacts (đã lưu) — tải về

* preprocessing: [here](https://drive.google.com/file/d/1Ea4Vu8Tn_w_buWuPhX0AG9SyCEaf_o8a/view?usp=sharing)
* scaler_target: [here](https://drive.google.com/file/d/1mxfsw5o5RcavgJ_8x98h4EkJX0IkNI6H/view?usp=sharing)

## Chạy local
Yêu cầu: Python 3.10+ (khuyến nghị)

```bash
cd Bird_Trace
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
uvicorn app:app --reload
````

Mở: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## API

* `GET /api/health`: trạng thái model + preprocessing artifacts
* `GET /api/sample`: dữ liệu mẫu 48h
* `POST /api/predict`: chạy dự đoán

Gợi ý: nếu dùng FastAPI, có thể kiểm tra OpenAPI tại `/docs` để đảm bảo README khớp schema request/response.