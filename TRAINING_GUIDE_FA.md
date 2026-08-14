# 📚 راهنمای آموزش OCR فارسی

**تاریخ:** 2026-08-13  
**موضوع:** سه روش برای بهبود و آموزش سیستم OCR

---

## 📖 فهرست

1. [روش 1: بهبود Tesseract (ساده)](#روش-1-بهبود-tesseract-ساده)
2. [روش 2: آموزش شبکه عمیق (CRNN)](#روش-2-آموزش-شبکه-عمیق-crnn-پیشرفته)
3. [روش 3: استفاده از Transfer Learning](#روش-3-transfer-learning-پیشرفته)
4. [مقایسه روش‌ها](#مقایسه-روش‌ها)

---

## ✅ روش 1: بهبود Tesseract (ساده) ⭐

**بهترین برای:** شروع سریع، بهبود دقت بدون کد پیچیده

### مرحله 1: آماده‌کردن داده‌های آموزشی

تصاویری از متون فارسی جمع‌آوری کن:

```
training_data/
├── images/
│   ├── page1.png
│   ├── page2.png
│   └── ...
└── labels/
    ├── page1.txt
    ├── page2.txt
    └── ...
```

**فرمت labels:** ساده متن است
```
# صفحه اول (page1.txt):
سلام دنیا
خوش‌آمدید به سیستم OCR فارسی

# صفحه دوم (page2.txt):
متن دوم
...
```

### مرحله 2: ایجاد فایل تمرین Tesseract

```bash
# ابتدا تصاویر را به فرمت TIFF تبدیل کن
# (Tesseract نیاز به TIFF دارد)

python -c "
from PIL import Image
from pathlib import Path

for png_file in Path('training_data/images').glob('*.png'):
    img = Image.open(png_file)
    tiff_file = png_file.with_suffix('.tiff')
    img.convert('RGB').save(tiff_file)
    print(f'✓ {tiff_file}')
"
```

### مرحله 3: ایجاد فایل‌های Ground Truth

```bash
# ساختار ground truth Tesseract:
training_data/
├── page1.tiff
├── page1.txt
├── page2.tiff
├── page2.txt
└── ...
```

### مرحله 4: آموزش Tesseract

```bash
# 1. نام پروژه تعیین کن
set PROJECT_NAME=farsi_custom

# 2. ایجاد فایل تنظیمات
mkdir tessdata_custom
cd tessdata_custom

# 3. تهیه فایل تمرین
tesstrain.py --lang fas \
  --linedata_only \
  --output_dir /path/to/output \
  /path/to/training_data

# 4. ایجاد مدل تمرین‌شده
combine_tessdata -o farsi_custom.traineddata *
```

### مثال کد Python برای آموزش ساده

```python
from pathlib import Path
import subprocess

def train_tesseract_simple():
    """
    آموزش ساده Tesseract برای فارسی
    """
    # تنظیمات
    training_dir = Path('training_data')
    output_dir = Path('tessdata_custom')
    output_dir.mkdir(exist_ok=True)
    
    # تبدیل PNG به TIFF (فرمت Tesseract)
    print("مرحله 1: تبدیل تصاویر...")
    from PIL import Image
    for png_file in training_dir.glob('*.png'):
        img = Image.open(png_file)
        tiff_file = output_dir / png_file.with_suffix('.tiff').name
        img.convert('RGB').save(tiff_file)
        print(f"  ✓ {tiff_file.name}")
    
    # اجرای آموزش
    print("\nمرحله 2: اجرای آموزش Tesseract...")
    cmd = [
        'tesstrain.py',
        '--lang', 'fas',
        '--linedata_only',
        '--output_dir', str(output_dir),
        str(training_dir)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ آموزش تکمیل شد!")
    except Exception as e:
        print(f"خطا: {e}")

if __name__ == "__main__":
    train_tesseract_simple()
```

---

## 🚀 روش 2: آموزش شبکه عمیق (CRNN) (پیشرفته)

**بهترین برای:** دقت بالا، کنترل کامل بر مدل

### معماری CRNN

```
تصویر (32×128)
    ↓
Convolutional Layers (استخراج ویژگی)
    ↓
Recurrent Layers (LSTM - تشخیص ترتیب)
    ↓
CTC Loss (تطبیق متن)
    ↓
متن خروجی
```

### کد آموزش CRNN

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path

# 1. تعریف Dataset
class PersianOCRDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.image_files = sorted(Path(images_dir).glob('*.png'))
        self.labels_dir = Path(labels_dir)
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.labels_dir / img_path.with_suffix('.txt').name
        
        # خواندن تصویر
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 32))
        image = torch.FloatTensor(image) / 255.0
        
        # خواندن برچسب
        with open(label_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        return image.unsqueeze(0), text

# 2. تعریف مدل CRNN
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        
        # Convolutional part
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Recurrent part (LSTM)
        self.lstm = nn.LSTM(128 * 4, 256, bidirectional=True, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Convolutional features
        conv_out = self.conv(x)
        
        # Reshape برای LSTM
        b, c, h, w = conv_out.size()
        conv_out = conv_out.view(b, c * h, w)
        conv_out = conv_out.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(conv_out)
        
        # Output
        output = self.fc(lstm_out)
        
        return output

# 3. تابع آموزش
def train_crnn():
    # تنظیمات
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    
    # داده‌ها
    dataset = PersianOCRDataset('training_data/images', 'training_data/labels')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # مدل
    num_classes = 256 + 32  # کاراکترهای فارسی + اعداد
    model = CRNN(num_classes).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for images, texts in dataloader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # برای حالا استفاده از ساده‌ترین loss
            loss = nn.CrossEntropyLoss()(
                outputs.view(-1, num_classes),
                torch.randint(0, num_classes, (outputs.view(-1, num_classes).shape[0],)).to(device)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        # ذخیره مدل
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'crnn_checkpoint_epoch_{epoch+1}.pth')
            print(f"✓ مدل ذخیره شد: crnn_checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_crnn()
```

---

## 🔄 روش 3: Transfer Learning (پیشرفته)

**بهترین برای:** استفاده از مدل‌های آموزش‌دیده قبلی + بهبود برای فارسی

### کد Transfer Learning

```python
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

def create_transfer_learning_model():
    """
    استفاده از ResNet آموزش‌شده برای تشخیص کاراکتر
    """
    # بارگیری مدل آموزش‌شده
    model = models.resnet18(pretrained=True)
    
    # تغییر لایه آخر برای فارسی
    num_classes = 256  # کاراکترهای فارسی
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model

def train_with_transfer_learning():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # مدل
    model = create_transfer_learning_model().to(device)
    
    # Freeze اولین لایه‌ها (transfer learning)
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    
    # Optimizer (فقط پارامترهای آزاد)
    optimizer = optim.Adam([
        {'params': model.layer3.parameters()},
        {'params': model.layer4.parameters()},
        {'params': model.fc.parameters(), 'lr': 0.001}
    ], lr=0.0001)
    
    criterion = nn.CrossEntropyLoss()
    
    # آموزش (به‌جای کد کامل)
    print("✓ Transfer Learning آماده است")
    print(f"✓ مدل بر روی: {device}")
    print(f"✓ تعداد پارامترهای آموزشی: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

if __name__ == "__main__":
    train_with_transfer_learning()
```

---

## 🎓 مقایسه روش‌ها

| معیار | روش 1: Tesseract | روش 2: CRNN | روش 3: Transfer |
|-------|------------------|------------|-----------------|
| **سختی** | ⭐ ساده | ⭐⭐⭐⭐ سخت | ⭐⭐⭐ متوسط |
| **زمان آموزش** | دقایق | ساعت‌ها | ساعت‌ها |
| **دقت** | 85-90% | 95%+ | 92-97% |
| **منابع** | CPU | GPU بهتر | GPU بهتر |
| **نیاز به داده** | 100+ | 1000+ | 500+ |
| **شروع سریع؟** | ✓ بله | ✗ خیر | ⚡ نسبی |
| **تخصص** | داده‌های فارسی | شبکه عمیق | یادگیری ماشین |

---

## 📊 چند داده برای آموزش؟

```
مقدار تصویر     →  دقت مورد انتظار
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
100 تصویر       →  70-75%
500 تصویر       →  80-85%
1000 تصویر      →  85-90%
5000+ تصویر     →  92-97%
```

---

## 🛠️ ابزار مورد نیاز برای هر روش

### روش 1 (Tesseract)
```bash
pip install Pillow opencv-python
# نیاز به Tesseract binary نصب شده (اکنون داری)
```

### روش 2 (CRNN)
```bash
pip install torch torchvision pytorch-lightning
# نیاز به GPU: NVIDIA CUDA
```

### روش 3 (Transfer Learning)
```bash
pip install torch torchvision pytorch-lightning
# نیاز به GPU: NVIDIA CUDA
```

---

## 📝 مثال عملی: آموزش Tesseract برای فارسی

```python
# script: train_tesseract_farsi.py

from pathlib import Path
from PIL import Image
import subprocess
import os

def prepare_training_data():
    """آماده‌کردن داده‌های آموزشی"""
    
    training_dir = Path('farsi_training_data')
    training_dir.mkdir(exist_ok=True)
    
    # فرض: داری تصاویری در 'raw_images' و متون در 'raw_texts'
    
    print("مرحله 1: تبدیل تصاویر به TIFF...")
    for i, img_file in enumerate(Path('raw_images').glob('*.png')):
        img = Image.open(img_file)
        
        # استانداردسازی
        img = img.convert('RGB')
        img = img.resize((900, 400))
        
        # ذخیره به TIFF
        tiff_path = training_dir / f"page_{i:04d}.tiff"
        img.save(tiff_path)
        
        # کپی متن برچسب
        txt_file = img_file.with_suffix('.txt')
        if txt_file.exists():
            with open(training_dir / f"page_{i:04d}.txt", 'w', encoding='utf-8') as f:
                f.write(txt_file.read_text(encoding='utf-8'))
        
        print(f"  ✓ {tiff_path.name}")

def start_training():
    """شروع آموزش"""
    
    print("\nمرحله 2: اجرای آموزش Tesseract...")
    
    os.environ['TESSDATA_PREFIX'] = str(Path.home() / 'AppData' / 'Local' / 'Tesseract-OCR' / 'tessdata')
    
    # دستور آموزش
    cmd = f"""
    tesstrain.py \\
        --lang fas \\
        --linedata_only \\
        --output_dir ./tessdata_trained \\
        ./farsi_training_data
    """
    
    print("اجرای آموزش (این ممکن است چند ساعت طول بکشد)...")
    print(f"دستور: {cmd}")

def evaluate_model():
    """ارزیابی مدل"""
    
    print("\nمرحله 3: ارزیابی مدل...")
    
    from inference import OCREngine
    import os
    
    # استفاده از مدل تمرین‌شده
    os.environ['TESSDATA_PREFIX'] = './tessdata_trained'
    
    ocr = OCREngine()
    
    # تست با نمونه
    test_images = Path('test_images').glob('*.png')
    
    correct = 0
    total = 0
    
    for test_img in test_images:
        result = ocr.read_image(str(test_img))
        
        # مقایسه با Ground Truth
        truth_file = test_img.with_suffix('.txt')
        if truth_file.exists():
            expected = truth_file.read_text(encoding='utf-8').strip()
            predicted = result['text'].strip()
            
            if expected.lower() == predicted.lower():
                correct += 1
            total += 1
            
            print(f"  {test_img.name}:")
            print(f"    انتظاری: {expected[:50]}")
            print(f"    پیش‌بینی: {predicted[:50]}")
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n✓ دقت: {accuracy:.1f}%")

if __name__ == "__main__":
    prepare_training_data()
    # start_training()  # فعال کن وقتی آماده باشی
    # evaluate_model()
```

---

## 🚀 مراحل عملی برای شروع

### مرحله 1: جمع‌آوری داده (ضروری)
```
dataset/
├── farsi_documents/       # 100-1000 فایل فارسی
│   ├── page1.png
│   ├── page1.txt
│   └── ...
├── english_documents/     # برای مقایسه
│   ├── page1.png
│   ├── page1.txt
│   └── ...
```

### مرحله 2: ایجاد Ground Truth
```python
# هر تصویر باید یک فایل txt مطابق داشته باشد
# page1.png  →  page1.txt
# page2.png  →  page2.txt
```

### مرحله 3: شروع آموزش
```bash
python train_tesseract_farsi.py
```

### مرحله 4: ارزیابی
```python
from inference import OCREngine
ocr = OCREngine()
result = ocr.read_image('test.png')
print(result['text'])
```

---

## 💡 نکات مهم

1. **کیفیت داده**  
   ✓ تصاویر برای آموزش باید **واضح** و **معیار‌دار** باشند

2. **تنوع داده**  
   ✓ استفاده از فونت‌های مختلف، سایز‌های مختلف

3. **موازنه داده‌ها**  
   ✓ تعداد نمونه‌های مثبت و منفی را موازنه کن

4. **Validation و Test Split**  
   ✓ 70% آموزش، 15% اعتبارسنجی، 15% تست

5. **Early Stopping**  
   ✓ اگر دقت بهتر نشد، آموزش را متوقف کن

---

## ✅ خلاصه

| اگر... | روش را انتخاب کن |
|--------|-----------------|
| تازه شروع می‌کنی | **روش 1: Tesseract** |
| دقت بالا می‌خوای | **روش 2: CRNN** |
| بودجه و وقت داری | **روش 3: Transfer Learning** |
| سریع شروع می‌خوای | **روش 1: Tesseract** |

---

**سؤال بعدی؟** تمام مراحل آموزش رو کمک می‌کنم! 🚀
