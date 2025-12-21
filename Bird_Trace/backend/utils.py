from __future__ import annotations

import math
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

def floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)

def mercator_to_latlon(x: float, y: float) -> Tuple[float, float]:
    # Web Mercator (EPSG:3857) meters -> WGS84 lat/lon
    R = 6378137.0
    lon = (x / R) * 180.0 / math.pi
    lat = (2.0 * math.atan(math.exp(y / R)) - math.pi / 2.0) * 180.0 / math.pi
    return lat, lon

from pyproj import CRS, Transformer

WGS84 = CRS("EPSG:4326")
UTM28N = CRS("EPSG:32628")

# UTM -> WGS84
to_wgs84 = Transformer.from_crs(UTM28N, WGS84, always_xy=True)

def utm_to_latlon(x_m: float, y_m: float) -> tuple[float, float]:
    lat, lon = to_wgs84.transform(x_m, y_m)  # always_xy=True => input (x,y), output (lon,lat)
    return lat, lon

import numpy as np

def try_add_latlon_utm32628(df, x_col="x_m", y_col="y_m"):
    out = df.copy()
    if x_col not in out.columns or y_col not in out.columns:
        out["lat"] = np.nan
        out["lon"] = np.nan
        return out

    lats, lons = [], []
    for x, y in zip(out[x_col].to_numpy(), out[y_col].to_numpy()):
        lat, lon = utm_to_latlon(float(x), float(y))
        lats.append(lat)
        lons.append(lon)

    out["lat"] = lats
    out["lon"] = lons
    return out

def compute_cyclical_time_features(ts: pd.Series) -> pd.DataFrame:
    dt_index = pd.to_datetime(ts)
    hour = dt_index.dt.hour.astype(float)
    day = dt_index.dt.day.astype(float)
    month = dt_index.dt.month.astype(float)

    sin_hour = np.sin(2*np.pi*hour/24.0)
    cos_hour = np.cos(2*np.pi*hour/24.0)
    sin_day = np.sin(2*np.pi*day/30.0)
    cos_day = np.cos(2*np.pi*day/30.0)
    sin_month = np.sin(2*np.pi*month/12.0)
    cos_month = np.cos(2*np.pi*month/12.0)

    return pd.DataFrame({
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "sin_day": sin_day,
        "cos_day": cos_day,
        "sin_month": sin_month,
        "cos_month": cos_month,
    })

def compute_step_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dist = np.sqrt(dx*dx + dy*dy)
    dist[0] = 0.0
    return dist

def parse_user_datetime(s: str) -> datetime:
    s = (s or "").strip()
    if not s:
        raise ValueError("datetime is empty")

    try:
        return datetime.fromisoformat(s.replace("Z",""))
    except Exception:
        pass

    for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue

    raise ValueError(f"Unsupported datetime format: {s}")
