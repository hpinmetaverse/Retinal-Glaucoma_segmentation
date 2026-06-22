"""Project configuration, seeding, and shared helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """Runtime configuration for training and inference."""

    data_root: Path = field(default_factory=lambda: PROJECT_ROOT / "dataset")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")

    train_subdir: str = "Training-20211018T055246Z-001/Training"
    test_subdir: str = "Test-20211018T060000Z-001/Test"

    targets: tuple[str, ...] = ("disc", "cup")
    target_display_names: tuple[str, ...] = ("Optic Disc", "Optic Cup")

    img_size: int = 256
    channels: int = 3
    batch_size: int = 4
    val_fraction: float = 0.20
    epochs: int = 50
    learning_rate: float = 1e-4
    mask_threshold: float = 0.5
    seed: int = 42

    model_filename: str = "unet_dristi_disc_cup.keras"
    test_metrics_filename: str = "test_metrics_disc_cup.csv"

    @property
    def train_dir(self) -> Path:
        return self.data_root / self.train_subdir

    @property
    def test_dir(self) -> Path:
        return self.data_root / self.test_subdir

    @property
    def train_image_dir(self) -> Path:
        return self.train_dir / "Images"

    @property
    def train_gt_dir(self) -> Path:
        return self.train_dir / "GT"

    @property
    def test_image_dir(self) -> Path:
        return self.test_dir / "Images"

    @property
    def test_gt_dir(self) -> Path:
        return self.test_dir / "Test_GT"

    @property
    def img_height(self) -> int:
        return self.img_size

    @property
    def img_width(self) -> int:
        return self.img_size

    @property
    def output_channels(self) -> int:
        return len(self.targets)

    @property
    def model_path(self) -> Path:
        return self.output_dir / self.model_filename

    @property
    def test_metrics_csv(self) -> Path:
        return self.output_dir / self.test_metrics_filename

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu() -> None:
    print("TensorFlow version:", tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("GPUs found:", gpus)
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                print("Could not set memory growth:", exc)
    else:
        print("No GPU found. Training will run on CPU.")


def make_disc_cup_overlay(image, disc_mask, cup_mask):
    overlay = image.copy()

    disc_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cup_color = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    disc_region = disc_mask > 0.5
    cup_region = cup_mask > 0.5

    overlay[disc_region] = 0.55 * overlay[disc_region] + 0.45 * disc_color
    overlay[cup_region] = 0.45 * overlay[cup_region] + 0.55 * cup_color
    return np.clip(overlay, 0.0, 1.0)
