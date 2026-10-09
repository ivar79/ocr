import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path


class CharsetManager:
    """Manager for character set used in OCR recognition."""
    
    def __init__(self):
        self.persian_chars = "آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیئؤ"
        self.persian_digits = "۰۱۲۳۴۵۶۷۸۹"
        self.english_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.english_digits = "0123456789"
        self.special_chars = " .,;:!?()-/\\\"'@#$%&*+=<>{}[]~`^|_"
        # ZWNJ (نیم‌فاصله) is a first-class token for Persian OCR
        self.persian_special = "،؛؟»«\u200c"
        
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
        """Return total number of classes including blank token."""
        return len(self.char_to_idx)


class OCRDetectionDataset(Dataset):
    """Dataset for text detection (CRAFT) training."""
    
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
    """Dataset for text recognition (CRNN) training."""
    
    def __init__(self, data_dir, charset_manager, transform=None,
                 target_height=32, max_width=512):
        self.data_dir = Path(data_dir)
        self.charset = charset_manager
        self.transform = transform
        self.target_height = target_height
        self.max_width = max_width
        self.samples = []
        self._widths = None
        
        label_file = self.data_dir / "labels.txt"
        if label_file.exists():
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        self.samples.append((parts[0], parts[1]))

    @property
    def widths(self):
        """Resized (target_height) widths, used for width-bucketed sampling."""
        if self._widths is None:
            self._widths = []
            for img_name, _ in self.samples:
                img_path = self.data_dir / "images" / img_name
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    self._widths.append(128)
                else:
                    h, w = img.shape[:2]
                    self._widths.append(int(round(w * self.target_height / max(h, 1))))
        return self._widths
                        
    def __len__(self):
        """Return actual number of samples (0 if empty)."""
        return len(self.samples)  # Fixed: removed max(1, ...)
        
    def __getitem__(self, idx):
        if not self.samples:
            return torch.zeros((1, 32, 128)), torch.tensor([0]), 1
            
        img_name, text = self.samples[idx]
        img_path = self.data_dir / "images" / img_name
        
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((32, 128), dtype=np.uint8)
        
        # Normalize polarity: dark text on a white background
        if np.mean(image) < 127:
            image = 255 - image
        
        # Aspect-preserving resize to fixed height (no squashing)
        h, w = image.shape[:2]
        scale = self.target_height / max(h, 1)
        target_w = max(8, int(round(w * scale)))
        target_w = min(target_w, self.max_width)
        image = cv2.resize(image, (target_w, self.target_height),
                           interpolation=cv2.INTER_LINEAR)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        
        # Encode text
        target = [self.charset.char_to_idx.get(c, 0) for c in text]
        target = torch.tensor(target, dtype=torch.long)
        
        return image, target, len(target)


class WidthBatchSampler(torch.utils.data.Sampler):
    """Batch samples of similar width together to minimize CTC padding.

    Samples are sorted by width; within chunks of ``4 * batch_size`` the
    order is shuffled each epoch so batches stay width-homogeneous but
    not identical across epochs.
    """

    def __init__(self, widths, batch_size, shuffle=True):
        self.widths = list(widths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.order = np.argsort(self.widths, kind="stable")

    def __iter__(self):
        order = self.order.copy()
        n = len(order)
        if self.shuffle:
            chunk = self.batch_size * 4
            chunks = [order[i:i + chunk] for i in range(0, n, chunk)]
            rng = np.random.default_rng()
            for ch in chunks:
                rng.shuffle(ch)
            order = np.concatenate(chunks) if chunks else order
        # Yield individual indices; DataLoader groups consecutive ones into
        # batches, so each batch stays width-homogeneous.
        yield from (int(x) for x in order)

    def __len__(self):
        return len(self.order)


def collate_fn(batch):
    """Custom collate function for variable-length targets (CTC).

    Images keep their (possibly different) widths; each batch is padded to
    the widest image so no aspect ratio is squashed.
    """
    images, targets, target_lengths = zip(*batch)
    max_w = max(img.size(2) for img in images)
    
    # Each image is (C, H, W); pad width to the widest sample
    padded = torch.zeros(len(images), images[0].size(0), images[0].size(1), max_w)
    for i, img in enumerate(images):
        padded[i, :, :, :img.size(2)] = img
    
    # Flatten targets
    targets = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return padded, targets, target_lengths
