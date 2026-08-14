import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path

class CharsetManager:
    def __init__(self):
        self.persian_chars = "آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"
        self.persian_digits = "۰۱۲۳۴۵۶۷۸۹"
        self.english_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.english_digits = "0123456789"
        self.special_chars = " .,;:!?()-/\\\"'@#$%&*+=<>{}[]~`^|_"
        self.persian_special = "،؛؟»«"
        
        self.full_charset = (
            self.persian_chars + self.persian_digits +
            self.english_chars + self.english_digits +
            self.special_chars + self.persian_special
        )
        
        # CTC blank token
        self.blank_token = "[BLANK]"
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.full_charset)}
        self.char_to_idx[self.blank_token] = 0
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
    
    @property
    def num_classes(self):
        return len(self.char_to_idx)

class OCRDetectionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = list(self.data_dir.glob("images/*.jpg"))
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((640, 640, 3), dtype=np.uint8)
            
        # In a real dataset, we'd load the affinity and region maps here.
        # For now, we generate dummy targets
        h, w = image.shape[:2]
        region_map = np.zeros((h//2, w//2), dtype=np.float32)
        affinity_map = np.zeros((h//2, w//2), dtype=np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=region_map)
            image = augmented['image']
            region_map = augmented['mask']
            
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        region_map = torch.from_numpy(region_map).unsqueeze(0)
        affinity_map = torch.from_numpy(affinity_map).unsqueeze(0)
        
        return image, region_map, affinity_map

class OCRRecognitionDataset(Dataset):
    def __init__(self, data_dir, charset_manager, transform=None):
        self.data_dir = Path(data_dir)
        self.charset = charset_manager
        self.transform = transform
        self.samples = []
        
        label_file = self.data_dir / "labels.txt"
        if label_file.exists():
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        self.samples.append((parts[0], parts[1]))
                        
    def __len__(self):
        return max(1, len(self.samples))
        
    def __getitem__(self, idx):
        if not self.samples:
            return torch.zeros((1, 32, 128)), torch.tensor([0]), 1
            
        img_name, text = self.samples[idx]
        img_path = self.data_dir / "images" / img_name
        
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((32, 128), dtype=np.uint8)
            
        # Resize/Pad to 32x128
        image = cv2.resize(image, (128, 32))
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        
        # Encode text
        target = [self.charset.char_to_idx.get(c, 0) for c in text]
        target = torch.tensor(target, dtype=torch.long)
        
        return image, target, len(target)

def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images)
    
    # Flatten targets
    targets = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return images, targets, target_lengths
