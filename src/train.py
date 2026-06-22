"""Model training, callbacks, and history visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import tensorflow as tf

from src.losses import bce_dice_loss
from src.metrics import (
    dice_coef,
    iou_coef,
    optic_cup_dice,
    optic_cup_iou,
    optic_disc_dice,
    optic_disc_iou,
)
from src.model import build_unet
from src.utils import Config


def compile_model(model, config: Config):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=bce_dice_loss,
        metrics=[
            dice_coef,
            iou_coef,
            optic_disc_dice,
            optic_cup_dice,
            optic_disc_iou,
            optic_cup_iou,
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
        ],
    )
    return model


def get_callbacks(config: Config):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.model_path),
            monitor="val_dice_coef",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef",
            mode="max",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train_model(train_ds, val_ds, config: Config):
    config.ensure_output_dir()
    model = build_unet(config)
    compile_model(model, config)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=get_callbacks(config),
    )

    print("Best model path:", config.model_path)
    return model, history


def plot_training_history(history, save_path=None, show=True):
    history_dict = history.history
    metrics_to_plot = [
        ("loss", "val_loss", "Loss"),
        ("dice_coef", "val_dice_coef", "Mean Dice"),
        ("optic_disc_dice", "val_optic_disc_dice", "Optic Disc Dice"),
        ("optic_cup_dice", "val_optic_cup_dice", "Optic Cup Dice"),
        ("iou_coef", "val_iou_coef", "Mean IoU"),
        ("binary_accuracy", "val_binary_accuracy", "Binary Accuracy"),
    ]

    plt.figure(figsize=(15, 12))
    for idx, (train_key, val_key, title) in enumerate(metrics_to_plot, start=1):
        plt.subplot(3, 2, idx)
        if train_key in history_dict:
            plt.plot(history_dict[train_key], label="train")
        if val_key in history_dict:
            plt.plot(history_dict[val_key], label="validation")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved training history plot to:", save_path)

    if show:
        plt.show()
    else:
        plt.close()
