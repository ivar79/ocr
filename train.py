import argparse
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
from utils.dataset import CharsetManager, OCRDetectionDataset, OCRRecognitionDataset, collate_fn
from utils.metrics import character_error_rate, word_error_rate

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.txt"),
        logging.StreamHandler()
    ]
)

def set_seed(seed):
    import random
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

def train_detection(config):
    logging.info("Starting Detection (CRAFT) Training")
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    
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
        
    scaler = torch.cuda.amp.GradScaler(enabled=True)
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
            
            with torch.cuda.amp.autocast(enabled=True):
                region_preds, affinity_preds = model(images)
                loss = criterion(region_preds, region_targets) + criterion(affinity_preds, affinity_targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    charset = CharsetManager()
    model = CRNN(img_height=32, num_channels=1, num_classes=charset.num_classes).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get("epochs", 100))
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    try:
        train_dataset = OCRRecognitionDataset("data/train", charset)
        train_loader = DataLoader(
            train_dataset, batch_size=config.get("batch_size", 32), 
            shuffle=True, collate_fn=collate_fn
        )
    except Exception as e:
        logging.warning(f"Could not load dataset: {e}")
        return
        
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    epochs = config.get("epochs", 100)
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, targets, target_lengths in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=True):
                preds = model(images)
                batch_size = images.size(0)
                pred_lengths = torch.full(size=(batch_size,), fill_value=preds.size(1), dtype=torch.long)
                preds = preds.log_softmax(2).permute(1, 0, 2) # T, B, C
                loss = criterion(preds, targets, pred_lengths, target_lengths)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "models/pretrained/crnn_weights.pth")
            logging.info("Saved best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["detection", "recognition"], required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    
    Path("models/pretrained").mkdir(parents=True, exist_ok=True)
    
    if args.mode == "detection":
        train_detection(config.get("training", {}))
    elif args.mode == "recognition":
        train_recognition(config.get("training", {}))
