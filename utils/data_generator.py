import os
import random
import argparse
import urllib.request
import zipfile
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

class PersianTextGenerator:
    def __init__(self):
        # A small sample list, in a real scenario we'd use a large corpus
        self.words = [
            "سلام", "ایران", "تهران", "دانشگاه", "پایتون", "برنامه‌نویسی",
            "هوش", "مصنوعی", "کتاب", "ماشین", "آموزش", "توسعه", "سیستم",
            "پردازش", "تصویر", "شبکه", "عصبی", "داده", "الگوریتم", "پروژه",
            "OCR", "Python", "PyTorch", "Deep", "Learning", "Data", "Science"
        ]
        
    def generate_random_text(self, min_words=1, max_words=5):
        num_words = random.randint(min_words, max_words)
        return " ".join(random.choices(self.words, k=num_words))

class SyntheticOCRDataGenerator:
    def __init__(self, font_dir="fonts"):
        self.font_dir = Path(font_dir)
        self.fonts = self._load_fonts()
        
    def _load_fonts(self):
        self.font_dir.mkdir(parents=True, exist_ok=True)
        vazir_url = "https://github.com/rastikerdar/vazirmatn/releases/download/v33.0.3/vazirmatn-v33.0.3.zip"
        vazir_zip = self.font_dir / "vazirmatn.zip"
        vazir_ttf = self.font_dir / "Vazirmatn-Regular.ttf"
        
        if not vazir_ttf.exists():
            print("Downloading Vazirmatn font...")
            urllib.request.urlretrieve(vazir_url, vazir_zip)
            with zipfile.ZipFile(vazir_zip, 'r') as zip_ref:
                zip_ref.extract("Vazirmatn-Regular.ttf", self.font_dir)
            
        return [str(vazir_ttf)]

    def _add_noise(self, image):
        noise_typ = random.choice(["gauss", "s&p", "none"])
        if noise_typ == "none":
            return image
            
        row, col, ch = image.shape
        if noise_typ == "gauss":
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = image + gauss * 255
            return np.clip(noisy, 0, 255).astype(np.uint8)
        elif noise_typ == "s&p":
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            out[tuple(coords)] = 255
            # Pepper
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            out[tuple(coords)] = 0
            return out

    def generate_image(self, text, width=256, height=64):
        bg_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        img = Image.new('RGB', (width, height), color=bg_color)
        d = ImageDraw.Draw(img)
        
        font_path = random.choice(self.fonts)
        font_size = random.randint(20, 40)
        font = ImageFont.truetype(font_path, font_size)
        
        # Get text bounding box
        bbox = d.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # Random position
        x = random.randint(10, max(11, width - text_w - 10))
        y = random.randint(5, max(6, height - text_h - 5))
        
        text_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
        
        # We might need arabic_reshaper and bidi for correct Persian rendering
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
        except ImportError:
            bidi_text = text # Fallback if libraries not installed
            
        d.text((x, y), bidi_text, font=font, fill=text_color)
        
        # Convert to cv2 image for augmentation
        cv_img = np.array(img)
        cv_img = cv_img[:, :, ::-1].copy() # RGB to BGR
        
        # Random rotation (-5 to 5 degrees)
        angle = random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        cv_img = cv2.warpAffine(cv_img, M, (width, height), borderValue=bg_color)
        
        # Random blur
        if random.random() > 0.5:
            k = random.choice([3, 5])
            cv_img = cv2.GaussianBlur(cv_img, (k, k), 0)
            
        # Add noise
        cv_img = self._add_noise(cv_img)
        
        return cv_img

def generate_dataset(output_dir, num_samples, split_ratio=(0.8, 0.1, 0.1)):
    output_dir = Path(output_dir)
    splits = ['train', 'val', 'test']
    
    for split in splits:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        
    text_gen = PersianTextGenerator()
    img_gen = SyntheticOCRDataGenerator()
    
    labels = {s: [] for s in splits}
    
    for i in range(num_samples):
        r = random.random()
        if r < split_ratio[0]: split = 'train'
        elif r < split_ratio[0] + split_ratio[1]: split = 'val'
        else: split = 'test'
            
        text = text_gen.generate_random_text()
        img = img_gen.generate_image(text)
        
        img_name = f"sample_{i:06d}.jpg"
        img_path = output_dir / split / 'images' / img_name
        cv2.imwrite(str(img_path), img)
        
        labels[split].append(f"{img_name}\t{text}")
        
        if (i+1) % 100 == 0:
            print(f"Generated {i+1}/{num_samples} images...")
            
    for split in splits:
        with open(output_dir / split / 'labels.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(labels[split]))
            
    print(f"Dataset generation complete in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic OCR Data Generator")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    args = parser.parse_args()
    
    generate_dataset(args.output, args.num_samples)
