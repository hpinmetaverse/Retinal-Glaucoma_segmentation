#!/usr/bin/env python3
"""CLI entry point for retinal glaucoma optic disc and cup segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset import prepare_datasets
from src.predict import (
    evaluate_each_test_image,
    evaluate_model,
    load_trained_model,
    predict_single_image,
    show_predictions,
)
from src.train import plot_training_history, train_model
from src.utils import PROJECT_ROOT, Config, configure_gpu, set_random_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run inference for DRISTI optic disc and cup segmentation."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "dataset",
        help="Root directory containing the extracted DRISTI dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for saved models, metrics, and plots.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--img-size", type=int, default=256, help="Input image size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the U-Net model.")
    train_parser.add_argument(
        "--history-plot",
        type=Path,
        default=None,
        help="Optional path to save the training history plot.",
    )
    train_parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the training history plot.",
    )

    subparsers.add_parser("evaluate", help="Evaluate the best saved model on val and test sets.")

    predict_parser = subparsers.add_parser("predict", help="Run inference on a single image.")
    predict_parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to a DRISTI-style fundus PNG image.",
    )
    predict_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary masks.",
    )
    predict_parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the prediction visualization.",
    )

    metrics_parser = subparsers.add_parser(
        "test-metrics",
        help="Compute per-image test metrics and save a CSV report.",
    )
    metrics_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary masks.",
    )

    return parser


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)

    set_random_seed(config.seed)
    configure_gpu()
    config.ensure_output_dir()

    print("DATA_ROOT:", config.data_root)
    print("OUTPUT_DIR:", config.output_dir)
    print("Targets:", config.target_display_names)

    if args.command == "train":
        data = prepare_datasets(config)
        _, history = train_model(data["train_ds"], data["val_ds"], config)
        plot_training_history(
            history,
            save_path=args.history_plot,
            show=not args.no_show,
        )
        return

    data = prepare_datasets(config)
    model = load_trained_model(config)

    if args.command == "evaluate":
        evaluate_model(model, data["val_ds"], data["test_ds"])
        show_predictions(model, data["val_ds"], count=4, threshold=0.5)
        show_predictions(model, data["test_ds"], count=4, threshold=0.5)
        return

    if args.command == "predict":
        predict_single_image(
            model,
            args.image,
            config,
            threshold=args.threshold,
            show=not args.no_show,
        )
        return

    if args.command == "test-metrics":
        test_metrics_df = evaluate_each_test_image(
            model,
            data["test_pairs"],
            config,
            threshold=args.threshold,
        )
        test_metrics_df.to_csv(config.test_metrics_csv, index=False)
        print("Saved per-image metrics to:", config.test_metrics_csv)
        print(test_metrics_df.head(10))
        print(test_metrics_df.describe(numeric_only=True))
        return


if __name__ == "__main__":
    main()
