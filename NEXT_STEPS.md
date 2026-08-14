# 🚀 دستورات تکمیل پروژه OCR — مخصوص Gemini 3.1 Pro

> **هدف:** این فایل شامل دستورات گام‌به‌گام برای تکمیل پروژه OCR فارسی-انگلیسی با PyTorch است.
> هر بخش یک پرامپت مستقل است که می‌توانی مستقیماً به Gemini 3.1 Pro بدهی.

---

## 📊 خلاصه وضعیت فعلی پروژه

### ✅ تکمیل‌شده (آماده)

| فایل | وضعیت | توضیح |
|:-----|:------:|:------|
| `preprocessing/image_processing.py` | ✅ | کلاس `ImagePreprocessor` — Grayscale, Denoise, CLAHE, Deskew |
| `preprocessing/pdf_processing.py` | ✅ | کلاس `PDFProcessor` — استخراج متن PDF دیجیتال + OCR اسکن |
| `postprocessing/text_formatter.py` | ✅ | کلاس `TextPostProcessor` — نرمال‌سازی فارسی، فاصله‌گذاری |
| `utils/dataset.py` | ✅ | کلاس `CharsetManager` — مجموعه کاراکتر فارسی/انگلیسی + CTC blank |
| `api/app.py` | ✅ | FastAPI با endpoint های `/ocr/image`, `/ocr/pdf`, `/health` |
| `ui.py` | ✅ | رابط Gradio برای آپلود تصویر/PDF |
| `requirements.txt` | ✅ | وابستگی‌ها نصب شده |

### ⏳ نیمه‌تمام (ساختار هست، مغز ندارد)

| فایل | مشکل |
|:-----|:------|
| `models/detection/craft.py` | معماری نوشته شده ولی **وزن Pretrained ندارد** + باگ skip connection |
| `models/recognition/crnn.py` | معماری نوشته شده ولی **وزن Pretrained ندارد** |
| `models/recognition/transformer_ocr.py` | معماری TrOCR ساده نوشته شده ولی **وزن ندارد** |
| `inference.py` | Pipeline ساختاردهی شده ولی **از DummyDetector/DummyRecognizer استفاده می‌کند** |

### ❌ خالی / انجام‌نشده

| فایل/پوشه | وضعیت |
|:-----------|:------:|
| `train.py` | **کاملاً خالی** — هیچ کدی ندارد |
| `configs/config.yaml` | **خالی** — هیچ تنظیماتی ندارد |
| `models/pretrained/` | **خالی** — هیچ فایل وزنی ندارد |
| `data/` (train/val/test/raw/processed) | **همه خالی** — هیچ داده‌ای ندارد |
| `tests/` | **خالی** — هیچ تستی ندارد |
| `notebooks/` | **خالی** |

---

## 📋 ترتیب اجرای دستورات

> **مهم:** دستورات را به ترتیب شماره‌گذاری شده اجرا کن. هر دستور مستقل است و می‌توانی آن را جداگانه به Gemini بدهی.

```
مرحله ۱ → config.yaml (پیش‌نیاز همه)
مرحله ۲ → اصلاح CRAFT (Detection)
مرحله ۳ → اصلاح CRNN (Recognition)
مرحله ۴ → Dataset و DataLoader
مرحله ۵ → train.py (آموزش)
مرحله ۶ → اتصال مدل‌های واقعی به inference.py
مرحله ۷ → تست‌ها
مرحله ۸ → بهینه‌سازی
```

---

## 🔷 مرحله ۱: ساخت فایل تنظیمات

### پرامپت:

```
فایل configs/config.yaml در پروژه OCR فارسی من خالی است.
مسیر پروژه: c:\Users\Hossein\Desktop\ocr

لطفاً یک فایل config.yaml کامل و حرفه‌ای بنویس که شامل موارد زیر باشد:

1. تنظیمات مدل Detection (CRAFT):
   - مسیر وزن‌ها (models/pretrained/craft_weights.pth)
   - اندازه ورودی (input_size: 640)
   - آستانه تشخیص (text_threshold: 0.7, link_threshold: 0.4, low_text: 0.4)

2. تنظیمات مدل Recognition (CRNN):
   - مسیر وزن‌ها (models/pretrained/crnn_weights.pth)
   - ابعاد تصویر ورودی (img_height: 32, img_width: 128)
   - hidden_size: 256
   - تعداد کانال (num_channels: 1)

3. تنظیمات آموزش (Training):
   - learning_rate: 0.001
   - batch_size: 32
   - epochs: 100
   - optimizer: Adam
   - scheduler: CosineAnnealingLR
   - مسیر دیتاست (data/train, data/val, data/test)
   - checkpoint_dir: models/pretrained/

4. تنظیمات پیش‌پردازش:
   - target_size: [640, 640]
   - denoise_strength: 10
   - clahe_clip_limit: 2.0

5. تنظیمات عمومی:
   - device: auto (cuda if available)
   - seed: 42
   - num_workers: 4
   - languages: [fa, en]

از فرمت YAML استاندارد استفاده کن و کامنت‌های توضیحی بنویس.
```

---

## 🔷 مرحله ۲: اصلاح و تکمیل مدل CRAFT (Text Detection)

### پرامپت:

```
فایل models/detection/craft.py در پروژه OCR فارسی من نیاز به اصلاح و تکمیل دارد.
مسیر پروژه: c:\Users\Hossein\Desktop\ocr

مشکلات فعلی:
1. معماری CRAFT فعلی Skip Connection ندارد — ویژگی‌های VGG مستقیماً وارد upconv1 می‌شوند بدون اینکه feature map های میانی ترکیب شوند. این باعث از دست رفتن اطلاعات ریزدانه می‌شود.
2. متد detect() برای inference وجود ندارد (فقط forward هست).
3. هیچ متد load_weights() ندارد.
4. پشتیبانی از Pretrained weights مدل CRAFT اصلی وجود ندارد.

لطفاً:
1. معماری CRAFT را اصلاح کن تا Skip Connection (مثل U-Net) بین لایه‌های VGG و Upsampling داشته باشد. Feature map ها را از stage های مختلف VGG16 استخراج کن و با upconv ترکیب کن.

2. یک متد detect(image) اضافه کن که:
   - تصویر numpy (grayscale یا BGR) بگیرد
   - آن را به tensor تبدیل و normalize کند
   - forward pass انجام دهد
   - از region_score و affinity_score، بounding box ها را استخراج کند
   - از cv2.connectedComponents یا findContours برای استخراج box ها استفاده کند
   - لیستی از bounding box ها برگرداند: [[x1,y1,x2,y2], ...]

3. یک متد load_weights(path) اضافه کن که وزن‌ها را از فایل .pth بارگذاری کند.

4. یک class method یا factory برای دانلود وزن‌های آموزش‌دیده CRAFT اضافه کن.
   لینک وزن‌ها: https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ
   (اگر مستقیم نمیشه، کد دانلود با gdown بنویس)

5. تنظیمات text_threshold, link_threshold, low_text را از config.yaml بخواند.

کلاس CharsetManager در utils/dataset.py موجود است.
فایل config.yaml در configs/config.yaml قرار دارد.
```

---

## 🔷 مرحله ۳: اصلاح و تکمیل مدل CRNN (Text Recognition)

### پرامپت:

```
فایل models/recognition/crnn.py در پروژه OCR فارسی من نیاز به تکمیل دارد.
مسیر پروژه: c:\Users\Hossein\Desktop\ocr

مشکلات فعلی مدل CRNN:
1. فقط forward() دارد، هیچ متد inference/recognize ندارد
2. CTC Decoding پیاده‌سازی نشده
3. متد load_weights وجود ندارد
4. nn.LSTM داخل nn.Sequential قرار گرفته که خروجی tuple می‌دهد و مشکل‌ساز است

لطفاً:
1. باگ LSTM داخل Sequential را رفع کن — LSTM مستقیماً به عنوان self.rnn1 تعریف شود.

2. یک متد recognize(image) اضافه کن که:
   - تصویر numpy ناحیه متن (crop شده) را بگیرد
   - آن را resize کند به img_height=32 با حفظ نسبت تصویر
   - به tensor تبدیل و normalize کند
   - forward pass اجرا کند
   - با CTC Greedy Decoding یا Beam Search، متن را رمزگشایی کند
   - متن فارسی/انگلیسی برگرداند

3. CTC Greedy Decoder بنویس:
   - خروجی softmax بگیرد
   - argmax بزند
   - کاراکترهای تکراری متوالی را حذف کند
   - blank token ها را حذف کند
   - با CharsetManager (از utils/dataset.py) ایندکس‌ها را به کاراکتر تبدیل کند

4. CTC Beam Search Decoder هم اضافه کن (اختیاری ولی مفید)

5. load_weights(path) بنویس

6. یک property به نام last_confidence اضافه کن که اطمینان آخرین شناسایی را نگه دارد.

کلاس CharsetManager در utils/dataset.py هست با این ساختار:
- char_to_idx: دیکشنری کاراکتر به ایندکس
- idx_to_char: دیکشنری ایندکس به کاراکتر
- blank token ایندکس 0
- num_classes: تعداد کل کلاس‌ها
```

---

## 🔷 مرحله ۴: ساخت Dataset و DataLoader

### پرامپت:

```
پروژه OCR فارسی من نیاز به Dataset و DataLoader دارد.
مسیر پروژه: c:\Users\Hossein\Desktop\ocr
فایل utils/dataset.py فعلاً فقط CharsetManager دارد.

لطفاً:

1. در همان فایل utils/dataset.py کلاس‌های زیر را اضافه کن:

   الف) کلاس OCRDetectionDataset(Dataset):
   - برای آموزش مدل CRAFT
   - تصاویر و label ها (region score map و affinity score map) را بخواند
   - Data Augmentation با albumentations (RandomBrightnessContrast, GaussNoise, Rotate, etc.)
   - فرمت label: تصویر ورودی + ماسک region + ماسک affinity

   ب) کلاس OCRRecognitionDataset(Dataset):
   - برای آموزش مدل CRNN
   - هر نمونه: تصویر crop شده‌ی یک کلمه/خط + متن label آن
   - تصاویر resize شوند به ارتفاع 32 با حفظ نسبت، padding تا عرض ثابت (128)
   - label ها با CharsetManager به ایندکس تبدیل شوند
   - Data Augmentation مناسب

   پ) تابع collate_fn سفارشی:
   - برای CRNN: مدیریت طول‌های متفاوت label (برای CTC Loss)

2. یک فایل جدید utils/data_generator.py بساز:
   - کلاس SyntheticDataGenerator که داده مصنوعی فارسی تولید کند
   - با استفاده از Pillow و فونت‌های فارسی، تصاویر متنی تولید کند
   - متن‌های رندوم فارسی و انگلیسی روی پس‌زمینه‌های مختلف رندر کند
   - نویز، چرخش، blur و تغییر نور اعمال کند
   - حداقل 1000 نمونه برای شروع

3. یک فایل utils/metrics.py بساز با:
   - character_error_rate(pred, target) — CER
   - word_error_rate(pred, target) — WER
   - accuracy(pred, target)
   - از کتابخانه editdistance استفاده کن (اضافه‌اش کن به requirements.txt)
```

---

## 🔷 مرحله ۵: نوشتن train.py (فایل آموزش)

### پرامپت:

```
فایل train.py در پروژه OCR فارسی من کاملاً خالی است.
مسیر پروژه: c:\Users\Hossein\Desktop\ocr

لطفاً یک فایل train.py کامل و حرفه‌ای بنویس که شامل موارد زیر باشد:

1. دو حالت آموزش:
   - python train.py --mode detection → آموزش مدل CRAFT
   - python train.py --mode recognition → آموزش مدل CRNN

2. برای حالت Detection (CRAFT):
   - بارگذاری OCRDetectionDataset از utils/dataset.py
   - Loss: MSE Loss (بین predicted maps و ground truth maps)
   - Optimizer: Adam با learning_rate از config.yaml
   - Scheduler: CosineAnnealingLR
   - ذخیره بهترین مدل در models/pretrained/craft_weights.pth
   - لاگ training loss و validation loss

3. برای حالت Recognition (CRNN):
   - بارگذاری OCRRecognitionDataset از utils/dataset.py
   - Loss: CTC Loss (torch.nn.CTCLoss)
   - collate_fn سفارشی برای مدیریت طول‌های مختلف
   - Optimizer: Adam
   - Scheduler: CosineAnnealingLR
   - ذخیره بهترین مدل در models/pretrained/crnn_weights.pth
   - لاگ CER و WER روی validation set (از utils/metrics.py)

4. ویژگی‌های عمومی:
   - خواندن تنظیمات از configs/config.yaml (با pyyaml)
   - پشتیبانی از GPU (cuda) و CPU
   - Mixed Precision Training (torch.cuda.amp) اختیاری
   - Early Stopping
   - Checkpoint saving/loading (resume training)
   - Progress bar با tqdm
   - لاگ نتایج در فایل training_log.txt
   - تنظیم seed برای reproducibility
   - argparse برای CLI

5. pyyaml و tqdm و editdistance را به requirements.txt اضافه کن

ساختار import ها:
- from models.detection.craft import CRAFT
- from models.recognition.crnn import CRNN
- from utils.dataset import CharsetManager, OCRDetectionDataset, OCRRecognitionDataset
- from utils.metrics import character_error_rate, word_error_rate
```

---

## 🔷 مرحله ۶: اتصال مدل‌های واقعی به Pipeline (inference.py)

### پرامپت:

```
فایل inference.py در پروژه OCR فارسی من از DummyDetector و DummyRecognizer استفاده می‌کند و عملاً کار نمی‌کند.
مسیر پروژه: c:\Users\Hossein\Desktop\ocr

وضعیت فعلی inference.py:
- _load_detector() → یک DummyDetector برمی‌گرداند که همیشه لیست خالی برمی‌گرداند
- _load_recognizer() → یک DummyRecognizer برمی‌گرداند که همیشه رشته خالی برمی‌گرداند
- _load_config() → دیکشنری خالی برمی‌گرداند
- _crop_region() → تصویر اصلی بدون crop برمی‌گرداند

لطفاً inference.py را کاملاً بازنویسی کن:

1. _load_config(path):
   - config.yaml را با pyyaml بخواند
   - مقادیر پیش‌فرض برای وقتی که config ناقص است

2. _load_detector():
   - مدل CRAFT واقعی را از models/detection/craft.py لود کند
   - وزن‌ها را از مسیر config بارگذاری کند
   - مدل را به device (cuda/cpu) منتقل کند
   - در حالت eval() قرار دهد

3. _load_recognizer():
   - مدل CRNN واقعی را از models/recognition/crnn.py لود کند
   - CharsetManager را از utils/dataset.py بسازد
   - وزن‌ها را لود کند
   - در حالت eval() قرار دهد

4. _crop_region(image, box):
   - box به فرمت [x1, y1, x2, y2] است
   - تصویر را crop کند
   - اگر box خارج از محدوده تصویر بود، clamp کند
   - تصویر crop شده را برگرداند

5. خطایابی و مدیریت خطا:
   - اگر وزن‌ها موجود نبود، warning بدهد و از Dummy استفاده کند (graceful fallback)
   - try/except برای هر مرحله از pipeline
   - لاگ مناسب با logging module

6. متد read_image را اصلاح کن:
   - اگر preprocessor تصویر grayscale برگرداند، آن را به فرمت مناسب مدل تبدیل کن

بقیه ساختار فایل (read, read_image, recognize) خوب است و فقط نیاز به اتصال مدل‌های واقعی دارد.
```

---

## 🔷 مرحله ۷: نوشتن تست‌ها

### پرامپت:

```
پوشه tests/ در پروژه OCR فارسی من خالی است.
مسیر پروژه: c:\Users\Hossein\Desktop\ocr

لطفاً تست‌های جامع بنویس:

1. tests/test_preprocessing.py:
   - تست ImagePreprocessor با تصاویر مصنوعی numpy
   - تست خروجی grayscale بودن
   - تست deskew با تصاویر چرخیده
   - تست با تصاویر خالی یا تمام‌سیاه

2. tests/test_pdf_processing.py:
   - تست PDFProcessor با یک PDF ساده (ساخت PDF مصنوعی با reportlab یا fitz)
   - تست تشخیص PDF دیجیتال vs اسکن‌شده

3. tests/test_models.py:
   - تست forward pass مدل CRAFT با تنسور رندوم
   - تست شکل خروجی (output shape)
   - تست forward pass مدل CRNN با تنسور رندوم
   - تست شکل خروجی CRNN

4. tests/test_postprocessing.py:
   - تست نرمال‌سازی فارسی (ک عربی → ک فارسی)
   - تست اصلاح فاصله‌گذاری
   - تست با رشته خالی

5. tests/test_dataset.py:
   - تست CharsetManager (تعداد کلاس‌ها، mapping صحیح)
   - تست encode/decode کاراکترها

6. tests/test_integration.py:
   - تست end-to-end OCREngine (با مدل‌های dummy)
   - تست با فرمت‌های مختلف فایل
   - تست خطای فرمت نامعتبر

از pytest استفاده کن.
pytest را به requirements.txt اضافه کن.
یک فایل conftest.py بساز با fixture های مشترک (sample images, sample PDFs).
```

---

## 🔷 مرحله ۸: بهینه‌سازی و تبدیل مدل

### پرامپت:

```
پروژه OCR فارسی من آماده بهینه‌سازی است.
مسیر پروژه: c:\Users\Hossein\Desktop\ocr

لطفاً:

1. یک فایل جدید optimize.py در ریشه پروژه بساز:

   الف) تبدیل به ONNX:
   - مدل CRAFT را به ONNX تبدیل کند
   - مدل CRNN را به ONNX تبدیل کند
   - Dynamic axes برای batch size و عرض تصویر
   - ذخیره در models/pretrained/craft.onnx و crnn.onnx

   ب) Quantization:
   - Dynamic Quantization با torch.quantization (INT8)
   - برای CRNN (Linear و LSTM)
   - مقایسه سرعت قبل و بعد

   پ) TorchScript:
   - تبدیل مدل‌ها به TorchScript با torch.jit.script
   - ذخیره در models/pretrained/craft_scripted.pt و crnn_scripted.pt

2. بنچمارک:
   - زمان inference هر روش (PyTorch, ONNX, TorchScript, Quantized)
   - مصرف حافظه هر روش
   - خروجی نتایج در یک جدول

3. Dockerfile بساز در ریشه پروژه:
   - Python 3.10 slim
   - نصب وابستگی‌های سیستمی (libgl, poppler-utils)
   - نصب requirements.txt
   - اجرا با uvicorn
   - EXPOSE 8000

اجرا: python optimize.py --mode onnx|quantize|torchscript|benchmark
```

---

## 🔷 (اختیاری) مرحله ۹: دانلود و استفاده از وزن‌های Pretrained

### پرامپت:

```
پروژه OCR فارسی من نیاز به وزن‌های از پیش آموزش‌دیده دارد تا بدون آموزش از صفر کار کند.
مسیر پروژه: c:\Users\Hossein\Desktop\ocr

لطفاً یک فایل scripts/download_weights.py بساز که:

1. وزن‌های CRAFT را دانلود کند:
   - منبع: مخزن رسمی CRAFT (https://github.com/clovaai/CRAFT-pytorch)
   - یا از Google Drive با gdown
   - ذخیره در models/pretrained/craft_weights.pth

2. وزن‌های یک مدل CRNN pretrained (مثلاً از مخزن‌های عمومی):
   - ذخیره در models/pretrained/crnn_weights.pth

3. اگر وزن‌ها قبلاً دانلود شده‌اند، skip کند

4. نوار پیشرفت دانلود نشان دهد (tqdm)

5. مهم: اگر ساختار وزن‌های دانلودشده با معماری فعلی ما تفاوت دارد:
   - یک mapping/conversion انجام بده
   - یا معماری ما را طوری تغییر بده که با وزن‌های رسمی سازگار باشد

6. در README.md دستور اجرا را مستند کن:
   python scripts/download_weights.py

اضافه کن: gdown را به requirements.txt
```

---

## 🔷 (اختیاری) مرحله ۱۰: ساخت داده مصنوعی فارسی

### پرامپت:

```
پروژه OCR فارسی من هیچ داده‌ای در پوشه data/ ندارد.
مسیر پروژه: c:\Users\Hossein\Desktop\ocr

لطفاً فایل utils/data_generator.py را بنویس (یا تکمیل کن) که:

1. کلاس PersianTextGenerator:
   - لیستی از کلمات و جملات رایج فارسی داشته باشد (حداقل 500 کلمه)
   - جملات تصادفی تولید کند
   - ترکیب فارسی-انگلیسی هم تولید کند

2. کلاس SyntheticOCRDataGenerator:
   - تصاویر متنی مصنوعی تولید کند
   - از فونت‌های مختلف استفاده کند (حداقل فونت‌های سیستمی)
   - پس‌زمینه‌های متنوع (سفید، خاکستری، بافت‌دار)
   - اعمال نویز (Gaussian, Salt&Pepper)
   - چرخش تصادفی (±5 درجه)
   - تغییر اندازه و فشردگی
   - Blur تصادفی
   - تغییرات نور و کنتراست

3. تابع generate_dataset(output_dir, num_samples, split_ratio):
   - num_samples تصویر تولید کند
   - تقسیم به train/val/test
   - ذخیره تصاویر در data/train/images/, data/val/images/, data/test/images/
   - ذخیره label ها در data/train/labels.txt, ... (فرمت: image_name\ttext)
   - گزارش آماری از داده تولیدشده

4. CLI:
   python utils/data_generator.py --num_samples 5000 --output data/

فونت‌های فارسی رایگان:
- Vazirmatn: https://github.com/rastikerdar/vazirmatn
- Shabnam: https://github.com/rastikerdar/shabnam-font
(اگر نصب نیستند، با urllib دانلود کن)
```

---

## ⚡ راهنمای اجرای سریع (Quick Start)

اگر می‌خواهی سریع‌ترین مسیر را طی کنی، این ترتیب را دنبال کن:

```
1. مرحله ۱  → config.yaml
2. مرحله ۹  → دانلود وزن‌های آماده
3. مرحله ۲  → اصلاح CRAFT
4. مرحله ۳  → اصلاح CRNN
5. مرحله ۶  → اتصال به inference.py
6. تست      → python ui.py → آپلود تصویر → بررسی خروجی
```

با این ۶ مرحله، یک OCR کار‌کننده خواهی داشت. بقیه مراحل برای حرفه‌ای‌تر کردن است.

---

> **نکته:** هر پرامپت بالا مستقل است. کافیست متن داخل بلوک ``` را کپی کنی و به Gemini 3.1 Pro بدهی. مطمئن شو که مسیر پروژه را هم بگی.
