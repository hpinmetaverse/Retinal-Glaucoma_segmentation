"""Prediction, evaluation, and visualization utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.dataset import load_image_and_masks
from src.metrics import get_custom_objects, numpy_dice_iou
from src.utils import Config, make_disc_cup_overlay


def load_trained_model(config: Config):
    return tf.keras.models.load_model(
        str(config.model_path),
        custom_objects=get_custom_objects(),
    )


def evaluate_model(model, val_ds, test_ds):
    validation_results = model.evaluate(val_ds, return_dict=True, verbose=1)
    test_results = model.evaluate(test_ds, return_dict=True, verbose=1)

    print("Validation results")
    for key, value in validation_results.items():
        print(f"  {key}: {value:.4f}")

    print("\nTest results")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")

    return validation_results, test_results


def show_disc_and_cup_samples(dataset, count=4, title="Optic Disc and Cup Masks", show=True):
    images, masks = next(iter(dataset))
    count = min(count, images.shape[0])

    plt.figure(figsize=(4 * count, 14))

    for i in range(count):
        image = images[i].numpy()
        disc_mask = masks[i, :, :, 0].numpy()
        cup_mask = masks[i, :, :, 1].numpy()
        overlay = make_disc_cup_overlay(image, disc_mask, cup_mask)

        plt.subplot(4, count, i + 1)
        plt.imshow(image)
        plt.title(f"Image {i + 1}")
        plt.axis("off")

        plt.subplot(4, count, count + i + 1)
        plt.imshow(disc_mask, cmap="gray")
        plt.title("Optic Disc Mask")
        plt.axis("off")

        plt.subplot(4, count, 2 * count + i + 1)
        plt.imshow(cup_mask, cmap="gray")
        plt.title("Optic Cup Mask")
        plt.axis("off")

        plt.subplot(4, count, 3 * count + i + 1)
        plt.imshow(overlay)
        plt.title("Overlay: Disc Red, Cup Green")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()


def show_predictions(model, dataset, count=4, threshold=0.5, show=True):
    images, masks = next(iter(dataset))
    predictions = model.predict(images, verbose=0)
    count = min(count, images.shape[0])

    plt.figure(figsize=(4 * count, 22))
    for i in range(count):
        image = images[i].numpy()
        true_disc = masks[i, :, :, 0].numpy()
        true_cup = masks[i, :, :, 1].numpy()
        pred_disc = predictions[i, :, :, 0] > threshold
        pred_cup = predictions[i, :, :, 1] > threshold
        pred_overlay = make_disc_cup_overlay(
            image, pred_disc.astype(np.float32), pred_cup.astype(np.float32)
        )

        plt.subplot(6, count, i + 1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(6, count, count + i + 1)
        plt.imshow(true_disc, cmap="gray")
        plt.title("True Disc")
        plt.axis("off")

        plt.subplot(6, count, 2 * count + i + 1)
        plt.imshow(pred_disc, cmap="gray")
        plt.title("Pred Disc")
        plt.axis("off")

        plt.subplot(6, count, 3 * count + i + 1)
        plt.imshow(true_cup, cmap="gray")
        plt.title("True Cup")
        plt.axis("off")

        plt.subplot(6, count, 4 * count + i + 1)
        plt.imshow(pred_cup, cmap="gray")
        plt.title("Pred Cup")
        plt.axis("off")

        plt.subplot(6, count, 5 * count + i + 1)
        plt.imshow(pred_overlay)
        plt.title("Pred Overlay")
        plt.axis("off")

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()


def evaluate_each_test_image(model, pairs, config: Config, threshold=0.5):
    rows = []
    for image_path, disc_mask_path, cup_mask_path in pairs:
        image, masks = load_image_and_masks(
            tf.constant(str(image_path)),
            tf.constant(str(disc_mask_path)),
            tf.constant(str(cup_mask_path)),
            config,
        )
        prediction = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]
        prediction_binary = (prediction > threshold).astype(np.float32)

        disc_dice, disc_iou = numpy_dice_iou(masks[:, :, 0].numpy(), prediction_binary[:, :, 0])
        cup_dice, cup_iou = numpy_dice_iou(masks[:, :, 1].numpy(), prediction_binary[:, :, 1])

        rows.append(
            {
                "image_id": image_path.stem,
                "label_folder": image_path.parent.name,
                "disc_dice": disc_dice,
                "disc_iou": disc_iou,
                "cup_dice": cup_dice,
                "cup_iou": cup_iou,
                "mean_dice": float(np.mean([disc_dice, cup_dice])),
                "mean_iou": float(np.mean([disc_iou, cup_iou])),
                "image_path": str(image_path),
                "disc_mask_path": str(disc_mask_path),
                "cup_mask_path": str(cup_mask_path),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_dice", ascending=True).reset_index(drop=True)


def predict_single_image(model, image_path, config: Config, threshold=0.5, show=True):
    image_path = Path(image_path)
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.image.decode_png(image_bytes, channels=config.channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [config.img_height, config.img_width], method="bilinear")

    prediction = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]
    disc_probability = prediction[:, :, 0]
    cup_probability = prediction[:, :, 1]
    disc_binary = disc_probability > threshold
    cup_binary = cup_probability > threshold
    overlay = make_disc_cup_overlay(
        image.numpy(), disc_binary.astype(np.float32), cup_binary.astype(np.float32)
    )

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(image.numpy())
    plt.title("Input")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(disc_probability, cmap="viridis")
    plt.title("Disc Probability")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(cup_probability, cmap="viridis")
    plt.title("Cup Probability")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(disc_binary, cmap="gray")
    plt.title("Disc Mask")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(cup_binary, cmap="gray")
    plt.title("Cup Mask")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(overlay)
    plt.title("Overlay: Disc Red, Cup Green")
    plt.axis("off")

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()

    return {
        "disc_probability": disc_probability,
        "cup_probability": cup_probability,
        "disc_mask": disc_binary,
        "cup_mask": cup_binary,
    }
