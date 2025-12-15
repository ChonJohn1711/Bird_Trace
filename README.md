# Bird Flight Path Prediction Demo (Web)

## Mục tiêu demo
- Input: 48h lịch sử (x_m, y_m + các yếu tố ảnh hưởng)
- Output: dự đoán vị trí 24h tới
- Có animation đường bay (marker chạy theo đường dự đoán)

Demo sẽ tự chọn chế độ hiển thị:
- Nếu `x_m/y_m` giống Web Mercator (|x|,|y| <= 20037508), frontend sẽ chuyển sang `lat/lon` và hiển thị trên bản đồ nền OpenStreetMap (cần internet).
- Nếu `x_m/y_m` trông giống `lat/lon` (độ), sẽ hiển thị trực tiếp.
- Nếu không khớp, fallback sang mặt phẳng XY (`CRS.Simple`).

## Cấu trúc
- `app.py`: FastAPI backend + serve static frontend
- `static/`: HTML/CSS/JS (Leaflet)
- `models/`: đặt model + scaler/encoder artifacts ở đây

## Chạy local
Yêu cầu: Python 3.10+ (khuyến nghị)

```bash
cd birdflight_demo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
uvicorn app:app --reload
```

Mở trình duyệt: http://127.0.0.1:8000

## Dùng model thật (joblib/pickle)
Backend sẽ cố gắng load model từ `models/` theo thứ tự ưu tiên:
- `*.joblib`
- `*.pkl`

Nếu inference lỗi, backend fallback sang heuristic predictor để demo vẫn chạy.

### Vì sao bạn gặp lỗi 288 vs 816 features?
Pipeline bạn gửi tạo input 48h dạng cửa sổ, sau đó **flatten**:
- Mỗi timestep có `features + target` = 15 + 2 = 17 cột
- INPUT_WINDOW = 48

Vì vậy input cho model ML cổ điển có kích thước:
- `48 * 17 = 816` features

Bản demo trước đó chỉ gửi 6 cột/timestep nên thành `48 * 6 = 288`.

### Thứ tự cột/timestep mà demo đang dùng (khớp pipeline bạn gửi)
Mỗi dòng lịch sử sẽ được chuẩn hóa về các cột dưới đây (theo đúng order khi bạn tạo `df[features + target]`):

1) `external_temperature`
2) `ground_speed`
3) `height_above_msl`
4) `gls_light_level`
5) `sin_heading`
6) `cos_heading`
7) `sin_hour`
8) `cos_hour`
9) `sin_day`
10) `cos_day`
11) `sin_month`
12) `cos_month`
13) `distance`
14) `time_of_day_code` (từ `time_of_day`)
15) `season_code` (từ `season`)
16) `x_m`
17) `y_m`

Sau đó demo lấy **48 dòng cuối**, padding nếu thiếu, rồi flatten thành `(1, 816)`.

### CSV upload
CSV có thể dùng header như dataset bạn đưa (dấu `-` và `:` đều được), ví dụ:
- `external-temperature`, `ground-speed`, `height-above-msl`, `gls:light-level`

Tối thiểu nên có:
- `timestamp`, `x_m`, `y_m`

Nếu thiếu các cột sin/cos, demo sẽ tự sinh từ `timestamp`:
- `sin_hour/cos_hour` với chu kỳ 24
- `sin_day/cos_day` với mẫu bạn đưa: chu kỳ 30
- `sin_month/cos_month` chu kỳ 12

Nếu thiếu `distance`, demo sẽ tự tính từ x_m/y_m.

### Scaling + LabelEncoding (để chạy đúng như lúc train)
Pipeline bạn gửi có:
- StandardScaler: `external-temperature`, `gls:light-level`, `distance`
- RobustScaler: `ground-speed`, `height-above-msl`
- StandardScaler cho target: `x_m`, `y_m`
- LabelEncoder: `time_of_day`, `season`

Để backend demo tái hiện đúng preprocess, bạn cần **lưu artifacts** sang thư mục `models/`.

Ví dụ (thêm vào notebook/train script của bạn):

```python
import joblib

# Lưu dict bạn đã tạo
joblib.dump(scalers_encoders, "models/preprocessing.joblib")

# Lưu riêng scaler target để inverse output
joblib.dump(scaler_target, "models/scaler_target.joblib")
```

Demo sẽ tự tìm và load:
- `models/preprocessing.joblib`
- `models/scaler_target.joblib`

Nếu không có 2 file này, demo vẫn chạy model nhưng sẽ **bỏ qua scaling/inverse** (và ghi chú rõ trong UI).

## API nhanh
- `GET /api/health`: xem models + trạng thái preprocessing artifacts
- `GET /api/sample`: dữ liệu mẫu 48h
- `POST /api/predict`: chạy dự đoán

