from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from .utils import compute_cyclical_time_features, compute_step_distance

@dataclass
class DatasetInfo:
    path: str
    rows: int
    start_ts: str
    end_ts: str
    columns: List[str]

class DataStore:
    def __init__(self, path: str):
        self.path = path
        self.df: pd.DataFrame = pd.DataFrame()
        self.info: Optional[DatasetInfo] = None

    def load(self) -> DatasetInfo:
        df = pd.read_csv(self.path)
        if "timestamp" not in df.columns:
            raise ValueError("Dataset missing column: timestamp")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        df = df.fillna(0)
        if "heading" in df.columns:
            df = df.drop(columns=["heading"])

        self.df = df
        self.info = DatasetInfo(
            path=self.path,
            rows=len(df),
            start_ts=str(df["timestamp"].min()),
            end_ts=str(df["timestamp"].max()),
            columns=list(df.columns),
        )
        return self.info

    def ensure_derived_columns(self) -> None:
        missing = [c for c in ["sin_hour","cos_hour","sin_day","cos_day","sin_month","cos_month"] if c not in self.df.columns]
        if missing:
            cyc = compute_cyclical_time_features(self.df["timestamp"])
            for c in cyc.columns:
                if c not in self.df.columns:
                    self.df[c] = cyc[c].to_numpy()

        if "distance" not in self.df.columns and {"x_m","y_m"}.issubset(self.df.columns):
            self.df["distance"] = compute_step_distance(self.df["x_m"].to_numpy(), self.df["y_m"].to_numpy())

    def get_window_strict(self, forecast_start: datetime, input_window: int) -> pd.DataFrame:
        history_end = forecast_start - timedelta(hours=1)
        history_start = forecast_start - timedelta(hours=input_window)

        df_idx = self.df.set_index("timestamp")
        grid = pd.date_range(start=history_start, end=history_end, freq="H")
        window = df_idx.reindex(grid)

        missing_rows = window[window.isna().any(axis=1)]
        if len(missing_rows) > 0:
            examples = [str(t) for t in missing_rows.index[:10]]
            raise ValueError(f"Missing required timestamps in history window. Examples: {examples}")

        window = window.reset_index().rename(columns={"index":"timestamp"})
        return window


    def get_window_last_records(self, forecast_start: datetime, input_window: int) -> pd.DataFrame:
        """Lenient window builder.

        Returns the last `input_window` rows strictly before `forecast_start` (sorted by timestamp).
        This avoids failing when the dataset has missing hours in the strict hourly grid.
        """
        if self.df is None or self.df.empty:
            raise ValueError("Dataset not loaded")

        df_before = self.df[self.df["timestamp"] < forecast_start].sort_values("timestamp")
        window = df_before.tail(input_window)
        if len(window) < input_window:
            earliest = str(self.df["timestamp"].min())
            latest = str(self.df["timestamp"].max())
            raise ValueError(
                f"Not enough history rows before forecast start. Need {input_window}, have {len(window)}. "
                f"Dataset range: {earliest} .. {latest}"
            )
        return window.reset_index(drop=True)

    def get_default_time(self) -> datetime:
        if self.df.empty:
            raise ValueError("Dataset not loaded")
        return self.df["timestamp"].max().to_pydatetime()
