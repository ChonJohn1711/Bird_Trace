from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib

@dataclass
class LoadedArtifacts:
    model_names: List[str]
    preprocessing: Optional[Dict[str, Any]]
    target_scaler: Optional[Any]

def list_model_files(models_dir: str) -> List[str]:
    if not os.path.isdir(models_dir):
        return []
    files = [fn for fn in os.listdir(models_dir) if fn.endswith(".joblib") and ("model" in fn.lower())]
    files.sort()
    return files

def load_artifacts(models_dir: str) -> LoadedArtifacts:
    preprocessing = None
    target_scaler = None

    pre_path = os.path.join(models_dir, "preprocessing.joblib")
    if os.path.exists(pre_path):
        preprocessing = joblib.load(pre_path)

    tgt_path = os.path.join(models_dir, "scaler_target.joblib")
    if os.path.exists(tgt_path):
        target_scaler = joblib.load(tgt_path)

    return LoadedArtifacts(
        model_names=list_model_files(models_dir),
        preprocessing=preprocessing,
        target_scaler=target_scaler,
    )

def load_model(models_dir: str, model_filename: str):
    path = os.path.join(models_dir, model_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
