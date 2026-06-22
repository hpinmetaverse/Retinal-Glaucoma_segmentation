# Retinal Glaucoma Segmentation

Production-ready TensorFlow implementation of a U-Net model for simultaneous optic disc and optic cup segmentation on retinal fundus images.
## Project Overview

This repository trains a multi-channel U-Net to segment two anatomical structures in retinal fundus images:

- **Optic disc** (output channel 0)
- **Optic cup** (output channel 1)

The model uses independent sigmoid outputs rather than softmax because the optic cup lies inside the optic disc. The workflow covers dataset pairing, stratified train/validation splitting, TensorFlow `tf.data` preprocessing, augmentation, training with callbacks, evaluation, and single-image inference.

## Features

- Multi-channel U-Net with encoder-decoder skip connections
- DRISTI-style image and soft-map mask pairing
- Stratified train/validation split by label folder (`GLAUCOMA` / `NORMAL`)
- TensorFlow input pipeline with caching, augmentation, and prefetching
- Combined BCE + Dice loss with Dice, IoU, and per-structure metrics
- Training callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`
- Validation and test evaluation
- Per-image test metrics exported to CSV
- Single-image prediction with overlay visualization
- CLI entry point for local training and inference

## Folder Structure

```text
Retinal-Glaucoma-Segmentation/
├── notebooks/
│   └── Advanced_UNET_Workshop.ipynb
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── dataset/                 # Place extracted DRISTI files here (not included)
├── outputs/                 # Saved models, plots, and CSV reports
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
└── main.py
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Retinal-Glaucoma-Segmentation
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The **DRISTI dataset is NOT included** in this repository. Download and extract it separately, then place the extracted contents inside the `dataset/` folder.

Expected layout after extraction:

```text
dataset/
├── Training-20211018T055246Z-001/
│   └── Training/
│       ├── Images/
│       │   ├── GLAUCOMA/
│       │   └── NORMAL/
│       └── GT/
│           └── <image_id>/
│               └── SoftMap/
│                   ├── <image_id>_ODsegSoftmap.png
│                   └── <image_id>_cupsegSoftmap.png
└── Test-20211018T060000Z-001/
    └── Test/
        ├── Images/
        └── Test_GT/
```

Example pairing:

- Image: `dataset/Training-.../Training/Images/GLAUCOMA/drishtiGS_002.png`
- Disc mask: `dataset/Training-.../Training/GT/drishtiGS_002/SoftMap/drishtiGS_002_ODsegSoftmap.png`
- Cup mask: `dataset/Training-.../Training/GT/drishtiGS_002/SoftMap/drishtiGS_002_cupsegSoftmap.png`

## Training

Train the model from the project root:

```bash
python main.py train
```

Optional arguments:

```bash
python main.py train --epochs 50 --batch-size 4 --img-size 256 --data-root ./dataset --output-dir ./outputs
```

The best model is saved to:

```text
outputs/unet_dristi_disc_cup.keras
```

For a quick dry run, use fewer epochs:

```bash
python main.py train --epochs 3
```

## Prediction

Evaluate the saved model on validation and test sets:

```bash
python main.py evaluate
```

Run single-image inference:

```bash
python main.py predict --image dataset/Test-.../Test/Images/GLAUCOMA/drishtiGS_101.png
```

Export per-image test metrics to CSV:

```bash
python main.py test-metrics
```

## Results

Training monitors:

- Loss (BCE + Dice)
- Mean Dice and IoU
- Optic disc Dice / IoU
- Optic cup Dice / IoU
- Binary accuracy

After training, evaluation reports validation and test metrics. The `test-metrics` command writes `outputs/test_metrics_disc_cup.csv` with per-image Dice and IoU scores for error analysis.

Typical outputs:

- `outputs/unet_dristi_disc_cup.keras` — best checkpoint by validation Dice
- `outputs/test_metrics_disc_cup.csv` — per-image test results

## Tech Stack

- Python 3.9+
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Pillow

## Notebook

An example workflow is available in `notebooks/Advanced_UNET_Workshop.ipynb`. It demonstrates how to use the modular `src/` package interactively. All executable logic lives in `src/`; the notebook is for exploration only.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
