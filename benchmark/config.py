from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Project layout
# -----------------------------------------------------------------------------
# Expected layout:
# ucf/
# ├── data/
# └── code/terrain_models/multitask_baselines/
#
# These paths are resolved relative to this file, so no machine-specific absolute
# paths are needed.
PACKAGE_DIR = Path(__file__).resolve().parent
TERRAIN_MODELS_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

SUBJECT_IDS = [1, 2, 7, 8, 10, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24]

VALID_TERRAINS = [
    "Concrete", "Grass", "Gravel", "DownStair",
    "Sand", "SlopeDown", "SlopeUp", "UpStair",
]

TERRAIN_ALIASES = {
    "slopdown": "SlopeDown", "slop_down": "SlopeDown", "slope_down": "SlopeDown",
    "slopup": "SlopeUp", "slop_up": "SlopeUp", "slope_up": "SlopeUp",
    "downstairs": "DownStair", "down_stair": "DownStair",
    "upstairs": "UpStair", "up_stair": "UpStair",
}

IMU_WINDOW_ROWS = 200
IMU_STEP_ROWS = 50
TRANSITION_SKIP_SEC = 0.30

VIDEO_NUM_FRAMES = 32
VIDEO_HEIGHT = 96
VIDEO_WIDTH = 96

TERRAIN_LOSS_WEIGHT = 1.0
TRANSITION_LOSS_WEIGHT = 1.0
REGRESSION_LOSS_WEIGHT = 1.0

DEFAULT_NUM_WORKERS = 4
DEFAULT_EPOCHS = 60
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 12
RANDOM_SEED = 42

THRESHOLD_VALUES = np.round(np.arange(0.10, 0.91, 0.05), 2)
VALIDATION_TOLERANCE_SEC = 0.50
EVALUATION_TOLERANCES_SEC = [0.25, 0.50, 1.00]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    video_kind: str | None
    feature_cols: tuple[str, ...]
    batch_size: int
    lstm_hidden_size: int
    lstm_num_layers: int
    lstm_dropout: float
    model_dim: int = 128
    dropout: float = 0.20

    @property
    def uses_video(self) -> bool:
        return self.video_kind is not None

    @property
    def cache_dir(self) -> Path:
        return DATA_ROOT / f"processed_loso_multitask_{self.name}"

    @property
    def sample_dir(self) -> Path:
        return self.cache_dir / "samples"

    @property
    def manifest_path(self) -> Path:
        return self.cache_dir / "multitask_manifest.csv"

    @property
    def label_path(self) -> Path:
        return self.cache_dir / "terrain_classes.json"

    @property
    def result_dir(self) -> Path:
        return TERRAIN_MODELS_DIR / f"results_loso_multitask_{self.name}"

    @property
    def checkpoint_dir(self) -> Path:
        return TERRAIN_MODELS_DIR / f"checkpoints_loso_multitask_{self.name}"


BASELINES = {
    "2imu": BaselineSpec(
        name="2imu",
        video_kind=None,
        feature_cols=(
            "ax0", "ay0", "az0", "gx0", "gy0", "gz0",
            "ax1", "ay1", "az1", "gx1", "gy1", "gz1",
        ),
        batch_size=256,
        lstm_hidden_size=128,
        lstm_num_layers=3,
        lstm_dropout=0.10,
        dropout=0.10,
    ),
    "1imu_shankcam": BaselineSpec(
        name="1imu_shankcam",
        video_kind="ShankCam",
        feature_cols=("ax0", "ay0", "az0", "gx0", "gy0", "gz0"),
        batch_size=16,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        lstm_dropout=0.20,
        dropout=0.20,
    ),
    "1imu_chestcam": BaselineSpec(
        name="1imu_chestcam",
        video_kind="ChestCam",
        feature_cols=("ax0", "ay0", "az0", "gx0", "gy0", "gz0"),
        batch_size=16,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        lstm_dropout=0.20,
        dropout=0.20,
    ),
}

ALIASES = {
    "shank": "1imu_shankcam",
    "chest": "1imu_chestcam",
    "two_imu": "2imu",
}


def get_spec(name: str) -> BaselineSpec:
    key = ALIASES.get(name.lower(), name.lower())
    if key not in BASELINES:
        raise ValueError(f"Unknown baseline {name!r}. Choose from: {', '.join(BASELINES)}")
    return BASELINES[key]
