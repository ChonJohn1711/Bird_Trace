from __future__ import annotations

import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    dataset_path: str = os.getenv("BIRD_DATASET_PATH", "data/data_preprocessing_WithoutNormalization.csv")
    models_dir: str = os.getenv("BIRD_MODELS_DIR", "models")
    default_input_window: int = int(os.getenv("BIRD_INPUT_WINDOW", "48"))
    default_horizon: int = int(os.getenv("BIRD_FORECAST_HORIZON", "24"))

settings = Settings()
