"""
Download and prepare Persian OCR Garshasp dataset for training.

This script downloads the Persian OCR Garshasp dataset from Hugging Face
and converts it to the project's format.

Usage:
    python scripts/download_garshasp_dataset.py --max-samples 100000
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm
import random

try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets library...")
    os.system("pip install datasets")
    from datasets import load_dataset


def download_and_convert_dataset(max_samples=None, val_split=0.1, test_split=0.1):
    """
    Download Garshasp dataset and convert to project format.

    Args:
        max_samples: Maximum number of samples to download (None for all)
        val_split: Fraction of data for validation
        test_split: Fraction of data for test
    """
    print("Loading Persian OCR Garshasp dataset from Hugging Face...")
    print("This may take a while depending on your internet connection...")

    # Load dataset
    dataset = load_dataset("AliShafiee2003/persian-ocr-garshasp", split="train")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Loaded {len(dataset)} samples")

    # Create directories
    base_dir = Path("data")
    train_dir = base_dir / "train" / "images"
    val_dir = base_dir / "val" / "images"
    test_dir = base_dir / "test" / "images"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=42)

    # Split dataset
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_test = int(n_total * test_split)
    n_train = n_total - n_val - n_test

    train_data = dataset.select(range(n_train))
    val_data = dataset.select(range(n_train, n_train + n_val))
    test_data = dataset.select(range(n_train + n_val, n_total))

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # Process and save training data
    print("\nProcessing training data...")
    train_labels = []
    for idx, sample in enumerate(tqdm(train_data, desc="Train")):
        # Save image
        image = sample["image"]
        image_path = train_dir / f"train_{idx:06d}.png"
        image.save(image_path)

        # Save label
        text = sample["text"]
        train_labels.append(f"train_{idx:06d}.png\t{text}")

    # Save training labels
    with open(base_dir / "train" / "labels.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_labels))

    # Process and save validation data
    print("\nProcessing validation data...")
    val_labels = []
    for idx, sample in enumerate(tqdm(val_data, desc="Val")):
        # Save image
        image = sample["image"]
        image_path = val_dir / f"val_{idx:06d}.png"
        image.save(image_path)

        # Save label
        text = sample["text"]
        val_labels.append(f"val_{idx:06d}.png\t{text}")

    # Save validation labels
    with open(base_dir / "val" / "labels.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val_labels))

    # Process and save test data
    print("\nProcessing test data...")
    test_labels = []
    for idx, sample in enumerate(tqdm(test_data, desc="Test")):
        # Save image
        image = sample["image"]
        image_path = test_dir / f"test_{idx:06d}.png"
        image.save(image_path)

        # Save label
        text = sample["text"]
        test_labels.append(f"test_{idx:06d}.png\t{text}")

    # Save test labels
    with open(base_dir / "test" / "labels.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(test_labels))

    print("\n✅ Dataset preparation complete!")
    print(f"Train: {len(train_labels)} samples")
    print(f"Val: {len(val_labels)} samples")
    print(f"Test: {len(test_labels)} samples")
    print(f"\nImages saved in:")
    print(f"  - {train_dir}")
    print(f"  - {val_dir}")
    print(f"  - {test_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare Persian OCR Garshasp dataset")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to download (None for all ~2.6M)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of data for test (default: 0.1)"
    )

    args = parser.parse_args()

    download_and_convert_dataset(
        max_samples=args.max_samples,
        val_split=args.val_split,
        test_split=args.test_split
    )
