"""Dataset discovery, splitting, and TensorFlow input pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils import Config

AUTOTUNE = tf.data.AUTOTUNE


def mask_suffix(mask_target):
    target = mask_target.lower().strip()
    if target == "disc":
        return "_ODsegSoftmap.png"
    if target == "cup":
        return "_cupsegSoftmap.png"
    raise ValueError("mask_target must be either 'disc' or 'cup'.")


def mask_path_for_target(image_path, gt_dir, target):
    image_path = Path(image_path)
    return gt_dir / image_path.stem / "SoftMap" / f"{image_path.stem}{mask_suffix(target)}"


def collect_image_mask_pairs(image_dir, gt_dir):
    image_paths = sorted(image_dir.rglob("*.png"))
    pairs = []
    missing = []

    for image_path in image_paths:
        disc_mask_path = mask_path_for_target(image_path, gt_dir, "disc")
        cup_mask_path = mask_path_for_target(image_path, gt_dir, "cup")

        if disc_mask_path.exists() and cup_mask_path.exists():
            pairs.append((image_path, disc_mask_path, cup_mask_path))
        else:
            if not disc_mask_path.exists():
                missing.append((image_path, disc_mask_path))
            if not cup_mask_path.exists():
                missing.append((image_path, cup_mask_path))

    if missing:
        print("Missing masks:")
        for image_path, expected_mask in missing[:10]:
            print("  Image:", image_path)
            print("  Expected mask:", expected_mask)
        raise FileNotFoundError(f"Missing {len(missing)} masks. Check dataset paths.")

    return pairs


def pair_table(pairs):
    rows = []
    for image_path, disc_mask_path, cup_mask_path in pairs:
        rows.append(
            {
                "image_id": image_path.stem,
                "label_folder": image_path.parent.name,
                "image_path": str(image_path),
                "disc_mask_path": str(disc_mask_path),
                "cup_mask_path": str(cup_mask_path),
            }
        )
    return pd.DataFrame(rows)


def stratified_train_val_split(pairs, val_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    groups = {}

    for pair in pairs:
        image_path = pair[0]
        label = image_path.parent.name.lower()
        groups.setdefault(label, []).append(pair)

    train_pairs = []
    val_pairs = []

    for label, items in sorted(groups.items()):
        items = list(items)
        rng.shuffle(items)
        if len(items) <= 1:
            n_val = 0
        else:
            n_val = max(1, int(round(len(items) * val_fraction)))
        val_pairs.extend(items[:n_val])
        train_pairs.extend(items[n_val:])
        print(f"{label}: train={len(items) - n_val}, validation={n_val}")

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    return train_pairs, val_pairs


def load_binary_mask(mask_path, config: Config):
    mask_bytes = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask_bytes, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.image.resize(mask, [config.img_height, config.img_width], method="nearest")
    mask = tf.cast(mask > config.mask_threshold, tf.float32)
    return mask


def load_image_and_masks(image_path, disc_mask_path, cup_mask_path, config: Config):
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_png(image_bytes, channels=config.channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [config.img_height, config.img_width], method="bilinear")

    disc_mask = load_binary_mask(disc_mask_path, config)
    cup_mask = load_binary_mask(cup_mask_path, config)
    masks = tf.concat([disc_mask, cup_mask], axis=-1)

    return image, masks


def augment_image_and_masks(image, masks):
    do_left_right = tf.random.uniform(()) > 0.5
    image = tf.cond(do_left_right, lambda: tf.image.flip_left_right(image), lambda: image)
    masks = tf.cond(do_left_right, lambda: tf.image.flip_left_right(masks), lambda: masks)

    do_up_down = tf.random.uniform(()) > 0.5
    image = tf.cond(do_up_down, lambda: tf.image.flip_up_down(image), lambda: image)
    masks = tf.cond(do_up_down, lambda: tf.image.flip_up_down(masks), lambda: masks)

    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    masks = tf.image.rot90(masks, k)

    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.90, upper=1.10)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, masks


def make_dataset(pairs, config: Config, training=False):
    image_paths = [str(pair[0]) for pair in pairs]
    disc_mask_paths = [str(pair[1]) for pair in pairs]
    cup_mask_paths = [str(pair[2]) for pair in pairs]

    ds = tf.data.Dataset.from_tensor_slices((image_paths, disc_mask_paths, cup_mask_paths))
    if training:
        ds = ds.shuffle(
            buffer_size=max(len(pairs), 1),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        lambda image_path, disc_mask_path, cup_mask_path: load_image_and_masks(
            image_path, disc_mask_path, cup_mask_path, config
        ),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.cache()

    if training:
        ds = ds.map(augment_image_and_masks, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(config.batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def prepare_datasets(config: Config):
    trainval_pairs = collect_image_mask_pairs(config.train_image_dir, config.train_gt_dir)
    test_pairs = collect_image_mask_pairs(config.test_image_dir, config.test_gt_dir)

    print(f"Training + validation image-disc-cup triples: {len(trainval_pairs)}")
    print(f"Test image-disc-cup triples: {len(test_pairs)}")

    train_pairs, val_pairs = stratified_train_val_split(
        trainval_pairs, config.val_fraction, config.seed
    )

    print("\nFinal split sizes")
    print("Train:", len(train_pairs))
    print("Validation:", len(val_pairs))
    print("Test:", len(test_pairs))

    train_ds = make_dataset(train_pairs, config, training=True)
    val_ds = make_dataset(val_pairs, config, training=False)
    test_ds = make_dataset(test_pairs, config, training=False)

    return {
        "trainval_pairs": trainval_pairs,
        "test_pairs": test_pairs,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "trainval_df": pair_table(trainval_pairs),
        "test_df": pair_table(test_pairs),
    }
