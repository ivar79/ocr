import argparse
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from models.detection.craft import CRAFT
from models.recognition.crnn import CRNN
from utils.dataset import (
    CharsetManager, OCRDetectionDataset, OCRRecognitionDataset,
    collate_fn, WidthBatchSampler,
)
from utils.metrics import character_error_rate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.txt"),
        logging.StreamHandler()
    ]
)


# ---------------------------------------------------------------------------
# AMP compatibility shim (torch.amp in modern torch, torch.cuda.amp in older)
# ---------------------------------------------------------------------------
def _make_amp():
    try:
        from torch.amp import autocast as new_autocast, GradScaler as new_scaler
        def autocast(enabled=True):
            return new_autocast('cuda', enabled=enabled)
        def scaler_factory():
            return new_scaler('cuda', enabled=True)
        return autocast, scaler_factory
    except ImportError:
        from torch.cuda.amp import autocast as old_autocast, GradScaler as old_scaler
        def autocast(enabled=True):
            return old_autocast(enabled=enabled)
        def scaler_factory():
            return old_scaler(enabled=True)
        return autocast, scaler_factory


autocast, GradScaler = _make_amp()


def set_seed(seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return {}


def resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.no_grad()
def evaluate_cer(model, loader, charset, device) -> float:
    """Average character error rate over a dataset using greedy decoding."""
    model.eval()
    total_cer = 0.0
    count = 0
    for images, targets, target_lengths in loader:
        images = images.to(device)
        preds = model(images)                     # (B, T, C)
        probs, indices = torch.max(torch.softmax(preds, dim=-1), dim=-1)
        indices = indices.cpu().numpy()
        # Decode targets
        flat_targets = targets.cpu().numpy()
        offset = 0
        for b in range(images.size(0)):
            text = ""
            prev = -1
            for idx in indices[b]:
                if idx != prev and idx != 0:
                    text += charset.idx_to_char.get(int(idx), "")
                prev = int(idx)
            tlen = int(target_lengths[b])
            target_text = "".join(
                charset.idx_to_char.get(int(c), "") for c in flat_targets[offset:offset + tlen])
            offset += tlen
            total_cer += character_error_rate(text, target_text)
            count += 1
    model.train()
    return total_cer / count if count else 1.0


def train_detection(config):
    logging.info("Starting Detection (CRAFT) Training")
    device = resolve_device(config.get("device", "auto"))
    use_amp = device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    model = CRAFT(pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get("epochs", 100))
    criterion = nn.MSELoss()

    # Dummy Dataset for now since OCRDetectionDataset might be empty
    try:
        train_dataset = OCRDetectionDataset("data/train")
        train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 8), shuffle=True)
    except Exception as e:
        logging.warning(f"Could not load dataset: {e}")
        return

    epochs = config.get("epochs", 100)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, region_targets, affinity_targets in pbar:
            images = images.to(device)
            region_targets = region_targets.to(device)
            affinity_targets = affinity_targets.to(device)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                region_preds, affinity_preds = model(images)
                loss = criterion(region_preds, region_targets) + criterion(affinity_preds, affinity_targets)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "models/pretrained/craft_weights.pth")
            logging.info("Saved best model.")


def train_recognition(config):
    logging.info("Starting Recognition (CRNN) Training")
    device = resolve_device(config.get("device", "auto"))
    use_amp = device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    logging.info(f"Using device: {device} (AMP: {use_amp})")

    charset = CharsetManager()
    hidden_size = config.get("hidden_size", 256)
    cnn_scale = config.get("cnn_scale", 1.0)
    resume_path = config.get("resume")
    if resume_path:
        model = CRNN.from_checkpoint(resume_path, map_location=str(device)).to(device)
        logging.info(f"Resumed from checkpoint: {resume_path} (arch: {model.arch()})")
    else:
        model = CRNN(img_height=32, num_channels=1, num_classes=charset.num_classes,
                     hidden_size=hidden_size, cnn_scale=cnn_scale).to(device)
        logging.info(f"CRNN arch: hidden={hidden_size} cnn_scale={cnn_scale}")

    optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get("epochs", 100))
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    data_dirs = config.get("data_dir", {}) or {}
    train_dir = Path(data_dirs.get("train", "data/train"))
    val_dir = Path(data_dirs.get("val", "data/val"))
    weights_out = Path(config.get("checkpoint_dir", "models/pretrained/")) / "crnn_weights.pth"
    weights_out.parent.mkdir(parents=True, exist_ok=True)

    batch_size = config.get("batch_size", 32)

    try:
        train_dataset = OCRRecognitionDataset(train_dir, charset)
        if len(train_dataset) == 0:
            logging.warning("Training dataset is empty.")
            return
        # Width-bucketed sampling: batches only pad to their widest member,
        # which keeps CTC cost proportional to actual line widths.
        train_sampler = WidthBatchSampler(train_dataset.widths, batch_size)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            shuffle=False, collate_fn=collate_fn, num_workers=0)
    except Exception as e:
        logging.warning(f"Could not load training dataset: {e}")
        return

    val_loader = None
    if val_dir.exists():
        val_dataset = OCRRecognitionDataset(val_dir, charset)
        if len(val_dataset) > 0:
            val_sampler = WidthBatchSampler(val_dataset.widths, batch_size, shuffle=False)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, sampler=val_sampler,
                shuffle=False, collate_fn=collate_fn, num_workers=0)

    epochs = config.get("epochs", 100)
    patience = config.get("patience", 8)
    min_epochs = config.get("min_epochs", 10)
    best_cer = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, targets, target_lengths in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                preds = model(images)
                batch_size = images.size(0)
                pred_lengths = torch.full(size=(batch_size,), fill_value=preds.size(1), dtype=torch.long)
                preds = preds.log_softmax(2).permute(1, 0, 2)  # T, B, C
                loss = criterion(preds, targets, pred_lengths, target_lengths)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        scheduler.step()

        # Validation
        val_cer = None
        if val_loader is not None:
            val_cer = evaluate_cer(model, val_loader, charset, device)
            logging.info(f"Epoch {epoch+1} - Val CER: {val_cer:.4f}")

            improved = val_cer < best_cer - 1e-6
            if improved:
                best_cer = val_cer
                epochs_without_improvement = 0
                model.save_checkpoint(str(weights_out))
                logging.info(f"Saved best model ({best_cer:.4f}) -> {weights_out}")
            elif epoch + 1 >= min_epochs:
                # Early stopping only counts once the model has had a chance
                # to start learning (CER often plateaus at 1.0 for the first
                # several epochs with a randomly-initialized CTC model).
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logging.info(f"Early stopping after {epoch+1} epochs (best val CER {best_cer:.4f})")
                    break
        else:
            # No validation set: track best training loss instead
            if avg_loss < best_cer:
                best_cer = avg_loss
                model.save_checkpoint(str(weights_out))
                logging.info(f"Saved best model (loss {avg_loss:.4f}) -> {weights_out}")

    logging.info(f"Training finished. Best val CER: {best_cer:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["detection", "recognition"], required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", default=None, help="Path to a CRNN checkpoint to warm-start from")
    parser.add_argument("--threads", type=int, default=None, help="torch intra-op threads (CPU only)")
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--cnn-scale", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = dict(config.get("training", {}))
    # Recognition architecture parameters (hidden_size, cnn_scale)
    training_cfg.update({k: v for k, v in (config.get("recognition") or {}).items()
                         if k in ("hidden_size", "cnn_scale")})

    # CLI overrides
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size
    if args.device is not None:
        training_cfg["device"] = args.device
    if args.lr is not None:
        training_cfg["learning_rate"] = args.lr
    if args.resume is not None:
        training_cfg["resume"] = args.resume
    if args.threads is not None:
        torch.set_num_threads(args.threads)
        logging.getLogger().info(f"torch threads set to {args.threads}")
    if args.hidden_size is not None:
        training_cfg["hidden_size"] = args.hidden_size
    if args.cnn_scale is not None:
        training_cfg["cnn_scale"] = args.cnn_scale

    set_seed(training_cfg.get("seed", 42))

    Path("models/pretrained").mkdir(parents=True, exist_ok=True)

    if args.mode == "detection":
        train_detection(training_cfg)
    elif args.mode == "recognition":
        train_recognition(training_cfg)
