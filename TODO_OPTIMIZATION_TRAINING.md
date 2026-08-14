# 📋 نقشه راه بهینه‌سازی و آموزش OCR فارسی

**نسخه:** 1.0  
**تاریخ:** 2026-08-13  
**وضعیت:** در حال اجرا 🚀

---

## 📊 خلاصه کلی

این سند تمام کارهای مورد نیاز برای بهینه‌سازی و آموزش سیستم OCR فارسی را شامل می‌شود. کارها به سه دسته تقسیم‌شده‌اند:

- ⚡ **فوری** (1-2 روز)
- 📈 **کوتاه‌مدت** (1-2 هفته)
- 🎯 **بلند‌مدت** (1-3 ماه)

---

## ⚡ کارهای فوری (Immediate - First Day)

### 1. ✅ آماده‌سازی اولیه سیستم
- [x] نصب Tesseract
- [x] دانلود مدل‌های فارسی و انگلیسی
- [x] فعال‌کردن رابط Gradio
- [x] تأیید کار سیستم OCR

**وضعیت:** ✅ **تکمیل‌شده**

---

### 2. تست سیستم با نمونه‌های واقعی

**کار:**
- [ ] تهیه 10 تصویر نمونه فارسی (اسکن‌شده یا عکس)
- [ ] تهیه 10 تصویر نمونه انگلیسی
- [ ] آپلود در رابط Gradio
- [ ] ثبت نتایج و خطاهای OCR

**فایل‌های مرتبط:**
- `ui.py` (رابط)
- `backend/ocr_backend.py`

**انتظار:** ~2 ساعت

---

### 3. ایجاد راهنمای استفاده سریع

**کار:**
- [x] نوشتن `QUICK_START_FA.md`
- [x] نوشتن `QUICK_START_COMMANDS.txt`
- [ ] ایجاد ویدیو آموزشی (اختیاری)

**وضعیت:** ✅ **تکمیل‌شده**

---

## 📈 کارهای کوتاه‌مدت (Short-term - 1-2 weeks)

### 1. بهبود کیفیت OCR برای فارسی

#### 1.1 تنظیم پارامترهای Tesseract

**کار:**
- [ ] آزمایش PSM های مختلف (0-13)
- [ ] مقایسه نتایج برای فارسی
- [ ] انتخاب بهترین PSM برای نوع متن
- [ ] ثبت نتایج در `optimization_results.txt`

**کد نمونه:**
```python
from backend.ocr_backend import TesseractOCRBackend
import cv2

# تست PSM های مختلف
for psm in range(0, 14):
    config = f'--psm {psm} --oem 3'
    # تست کن و نتایج ثبت کن
```

**انتظار:** 4-6 ساعت

---

#### 1.2 بهبود Preprocessing

**کار:**
- [ ] آزمایش تنظیمات CLAHE (contrast)
- [ ] تست kernel های مختلف برای morphology
- [ ] آزمایش threshold های مختلف
- [ ] بررسی تأثیر Denoise

**فایل‌های مرتبط:**
- `preprocessing/image_processing.py`

**کد:**
```python
import cv2

# تست مقادیر مختلف CLAHE
for clipLimit in [1.0, 2.0, 3.0, 4.0]:
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    # تست و مقایسه
```

**انتظار:** 6-8 ساعت

---

#### 1.3 بهبود Postprocessing

**کار:**
- [ ] اضافه‌کردن تصحیح‌کننده‌ی هجی فارسی
- [ ] بهبود نرمال‌سازی کاراکترهای فارسی/عربی
- [ ] اضافه‌کردن شناسایی الگوهای شناخته‌شده
- [ ] تست بر روی نمونه‌های واقعی

**فایل‌های مرتبط:**
- `postprocessing/text_formatter.py`

**پیاده‌سازی:**
```python
class TextPostProcessor:
    def spell_check_farsi(self, text):
        # استفاده از فرهنگ فارسی
        pass
    
    def normalize_farsi(self, text):
        # نرمال‌سازی کاملتر
        pass
```

**انتظار:** 8-10 ساعت

---

### 2. جمع‌آوری داده‌های آموزشی

#### 2.1 جمع‌آوری تصاویر فارسی

**کار:**
- [ ] دانلود 500-1000 تصویر متن فارسی
- [ ] تنوع سازی: فونت‌های مختلف، سایز‌های مختلف، کیفیت مختلف
- [ ] تقسیم‌بندی: 70% آموزش، 15% اعتبار، 15% تست

**منابع:**
- Wikipedia فارسی (download tools)
- Open Books (فارسی)
- اسناد دولتی (با اجازه)
- مقالات و کتاب‌های دیجیتالی

**انتظار:** 8-12 ساعت

---

#### 2.2 ایجاد Ground Truth

**کار:**
- [ ] ایجاد اسکریپت OCR تصاویر
- [ ] تصحیح دستی خروجی‌های OCR
- [ ] ایجاد برچسب‌های دقیق
- [ ] ذخیره‌سازی ساختار شده

**فرمت:**
```
training_data/
├── images/
│   ├── page_0001.png
│   ├── page_0002.png
│   └── ...
└── labels/
    ├── page_0001.txt
    ├── page_0002.txt
    └── ...
```

**انتظار:** 20-30 ساعت (دستی)

---

### 3. آموزش Tesseract

#### 3.1 نصب ابزار آموزش

**کار:**
- [ ] نصب `tesstrain`
- [ ] نصب وابستگی‌های Python
- [ ] بررسی متطلبات سیستم

**دستور:**
```bash
pip install tesstrain
```

**انتظار:** 1 ساعت

---

#### 3.2 آموزش مدل اولیه

**کار:**
- [ ] اجرای آموزش Tesseract
- [ ] نظارت بر فرآیند
- [ ] تسجیل Accuracy در طول آموزش
- [ ] ذخیره checkpoint ها

**دستور:**
```bash
tesstrain.py ^
  --lang fas ^
  --linedata_only ^
  --output_dir ./tessdata_trained ^
  ./farsi_training_data
```

**انتظار:** 4-8 ساعت (تا 24 ساعت)

---

### 4. ارزیابی و مقایسه

**کار:**
- [ ] ارزیابی مدل اولیه
- [ ] مقایسه با Tesseract پیش‌فرض
- [ ] محاسبه Accuracy، Precision، Recall
- [ ] تحلیل خطاهای رایج

**متریک‌ها:**
```
- Accuracy: مجموع دقیق / کل
- Character Error Rate (CER)
- Word Error Rate (WER)
- Confidence Score
```

**انتظار:** 4-6 ساعت

---

## 🎯 کارهای بلند‌مدت (Long-term - 1-3 months)

### 1. آموزش شبکه عمیق CRNN

#### 1.1 تهیه داده‌ها برای CRNN

**کار:**
- [ ] آماده‌کردن 5000-10000 نمونه
- [ ] تقسیم‌بندی Train/Val/Test
- [ ] ایجاد Data Loader در PyTorch
- [ ] تست Data Pipeline

**فایل:**
```python
# utils/dataset.py
class PersianCRNNDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        pass
```

**انتظار:** 20-30 ساعت

---

#### 1.2 پیاده‌سازی مدل CRNN

**کار:**
- [ ] تعریف معماری CRNN
- [ ] پیاده‌سازی Loss Function (CTC Loss)
- [ ] تنظیم Optimizer
- [ ] اضافه‌کردن Data Augmentation

**فایل:**
```python
# models/recognition/crnn.py
class CRNN(nn.Module):
    def __init__(self, num_classes):
        # Conv + RNN + FC
        pass
```

**انتظار:** 10-15 ساعت

---

#### 1.3 آموزش CRNN

**کار:**
- [ ] نوشتن Training Loop
- [ ] اضافه‌کردن Validation
- [ ] Early Stopping
- [ ] ذخیره Best Model

**انتظار:** 20-48 ساعت (بستگی به GPU دارد)

---

### 2. Detection: آموزش CRAFT

#### 2.1 تهیه داده‌های Detection

**کار:**
- [ ] جمع‌آوری تصاویر
- [ ] Annotation با Bounding Boxes
- [ ] تحویل فرمت Yolo/Pascal VOC
- [ ] Validation Set

**ابزار:**
- LabelImg
- Roboflow
- یا دستی

**انتظار:** 30-50 ساعت

---

#### 2.2 آموزش CRAFT

**کار:**
- [ ] آماده‌کردن داده‌ها
- [ ] Transfer Learning از Pre-trained
- [ ] آموزش Model
- [ ] Evaluation

**انتظار:** 20-40 ساعت

---

### 3. Fine-tuning و Optimization

#### 3.1 Hyperparameter Tuning

**کار:**
- [ ] Grid Search برای بهترین پارامترها
- [ ] تست Learning Rate های مختلف
- [ ] تعیین Batch Size بهینه
- [ ] تنظیم Regularization

**انتظار:** 10-20 ساعت

---

#### 3.2 Model Optimization

**کار:**
- [ ] Quantization (کاهش سایز)
- [ ] Pruning (کاهش پارامترها)
- [ ] تحویل به ONNX
- [ ] Inference Optimization

**انتظار:** 8-15 ساعت

---

### 4. Deployment

#### 4.1 استقرار API

**کار:**
- [ ] تنظیم Docker
- [ ] Deploy به سرور
- [ ] تنظیم Load Balancing
- [ ] Monitoring

**فایل:**
```dockerfile
FROM python:3.10
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "api.app:app"]
```

**انتظار:** 6-10 ساعت

---

#### 4.2 استقرار UI

**کار:**
- [ ] Deploy Gradio
- [ ] یا ایجاد Web UI (React/Vue)
- [ ] تنظیم SSL
- [ ] Domain + CDN

**انتظار:** 4-8 ساعت

---

## 📋 کارهای اختیاری (Optional)

### 1. تحسین‌های اضافی

- [ ] ایجاد Language Model برای Post-correction
- [ ] تجمیع متعدد مدل‌ها (Ensemble)
- [ ] اضافه‌کردن پشتیبانی زبان‌های دیگر (عربی، اردو)
- [ ] Real-time Processing
- [ ] GPU Acceleration

---

### 2. توثیق و آموزش

- [ ] نوشتن داکومنتیشن کامل API
- [ ] ایجاد Tutorial ویدیویی
- [ ] نوشتن Paper/Blog Post
- [ ] ایجاد Community

---

## 🔄 ترتیب اولویت توصیه‌شده

### مرحله 1: تست و اعتبار سازی (1-2 روز)
```
1. تست با نمونه‌های واقعی
2. شناسایی مشکلات
3. اولویت‌بندی بهبود‌ها
```

### مرحله 2: بهبود Tesseract (1-2 هفته)
```
1. تنظیم PSM و پارامترها
2. بهبود Preprocessing
3. بهبود Postprocessing
4. Evaluation
```

### مرحله 3: آموزش داده‌ها (2-3 هفته)
```
1. جمع‌آوری تصاویر
2. ایجاد Ground Truth
3. آموزش Tesseract
4. Evaluation
```

### مرحله 4: شبکه عمیق (4-8 هفته)
```
1. CRNN (Recognition)
2. CRAFT (Detection)
3. Integration
4. Deployment
```

---

## 📊 جدول بررسی کارها

### فوری (Immediate)

| شماره | کار | وضعیت | انجام شده | ساعت |
|-------|------|-------|----------|------|
| 1.1 | نصب و تنظیم | ✅ | ✅ | 4 |
| 1.2 | تست واقعی | ⬜ | - | 2 |
| 1.3 | راهنما | ✅ | ✅ | 3 |

---

### کوتاه‌مدت (Short-term)

| شماره | کار | وضعیت | انجام شده | ساعت |
|-------|------|-------|----------|------|
| 2.1.1 | PSM Tuning | ⬜ | - | 6 |
| 2.1.2 | Preprocessing | ⬜ | - | 8 |
| 2.1.3 | Postprocessing | ⬜ | - | 10 |
| 2.2 | جمع‌آوری داده | ⬜ | - | 12 |
| 2.3 | Ground Truth | ⬜ | - | 30 |
| 2.4 | Tesseract Training | ⬜ | - | 8 |
| 2.5 | Evaluation | ⬜ | - | 6 |

**کل:** ~80 ساعت (2-3 هفته)

---

### بلند‌مدت (Long-term)

| کار | ساعت |
|------|------|
| CRNN | 40-50 |
| CRAFT | 30-40 |
| Optimization | 15-20 |
| Deployment | 10-15 |

**کل:** ~100-150 ساعت (1-3 ماه)

---

## 🛠️ ابزار و منابع مورد نیاز

### نرم‌افزار
- [ ] Tesseract v5.4+
- [ ] Python 3.10+
- [ ] PyTorch (برای شبکه عمیق)
- [ ] OpenCV
- [ ] Git

### داده‌ها
- [ ] منابع متن فارسی
- [ ] ابزار Annotation
- [ ] داده‌های Ground Truth

### منابع آموزشی
- [ ] Tesseract Documentation
- [ ] PyTorch Tutorials
- [ ] OCR Papers
- [ ] Community Forums

---

## 📝 نکات مهم

### ✅ اجرای موفق

- ✓ شروع با داده‌های کوچک برای تست سریع
- ✓ ثبت تمام نتایج برای مقایسه
- ✓ استفاده از Version Control
- ✓ Backup تمام مدل‌ها

### ⚠️ مسائل احتمالی

- ⚠️ کیفیت داده‌های آموزشی اهمیت داره
- ⚠️ Overfitting در نمونه‌های کم
- ⚠️ Computational Resources
- ⚠️ Time Management

---

## 📞 راه‌های کمک

اگر در هر مرحله مشکل داشتی:

1. **مستندات:** `TRAINING_GUIDE_FA.md`, `README.md`
2. **Code Examples:** در هر فایل
3. **Community:** Forum، GitHub Issues
4. **Research:** Papers، Articles

---

## 🎯 خلاصه

| دوره | تعداد کار | ساعت | اولویت |
|------|-----------|------|--------|
| فوری | 3 | 9 | 🔴 بسیار زیاد |
| کوتاه‌مدت | 7 | ~80 | 🟠 زیاد |
| بلند‌مدت | 4 | ~120 | 🟡 متوسط |
| اختیاری | ∞ | - | 🟢 کم |

**مجموع:** 14 کار اصلی، ~200 ساعت (5-6 هفته تا 3 ماه)

---

## ✨ نتیجه نهایی

پس از اتمام این کارها، خواهی داشت:

✅ سیستم OCR فارسی بهینه‌شده  
✅ مدل آموزش‌شده Tesseract  
✅ شبکه عمیق CRNN  
✅ Detection System (CRAFT)  
✅ API و UI آماده برای استقرار  
✅ مستندات کامل و Community

---

**آخرین بروز‌رسانی:** 2026-08-13  
**نسخه:** 1.0  
**نویسنده:** OCR Project Team
