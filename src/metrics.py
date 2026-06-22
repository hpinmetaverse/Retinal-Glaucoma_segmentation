"""Segmentation metrics for training and evaluation."""

import numpy as np
import tensorflow as tf

SMOOTH = 1e-6


def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    denominator = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
    dice = (2.0 * intersection + SMOOTH) / (denominator + SMOOTH)
    return tf.reduce_mean(dice)


def iou_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2]) - intersection
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return tf.reduce_mean(iou)


def optic_disc_dice(y_true, y_pred):
    return dice_coef(y_true[..., 0:1], y_pred[..., 0:1])


def optic_cup_dice(y_true, y_pred):
    return dice_coef(y_true[..., 1:2], y_pred[..., 1:2])


def optic_disc_iou(y_true, y_pred):
    return iou_coef(y_true[..., 0:1], y_pred[..., 0:1])


def optic_cup_iou(y_true, y_pred):
    return iou_coef(y_true[..., 1:2], y_pred[..., 1:2])


def numpy_dice_iou(y_true, y_pred_binary, smooth=1e-6):
    y_true = y_true.astype(np.float32).reshape(-1)
    y_pred_binary = y_pred_binary.astype(np.float32).reshape(-1)

    intersection = np.sum(y_true * y_pred_binary)
    denominator = np.sum(y_true) + np.sum(y_pred_binary)
    union = denominator - intersection

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    iou = (intersection + smooth) / (union + smooth)
    return float(dice), float(iou)


def get_custom_objects():
    from src.losses import bce_dice_loss

    return {
        "bce_dice_loss": bce_dice_loss,
        "dice_coef": dice_coef,
        "iou_coef": iou_coef,
        "optic_disc_dice": optic_disc_dice,
        "optic_cup_dice": optic_cup_dice,
        "optic_disc_iou": optic_disc_iou,
        "optic_cup_iou": optic_cup_iou,
    }
