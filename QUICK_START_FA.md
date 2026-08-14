# 🚀 راهنمای شروع سریع - سیستم OCR فارسی

**نسخه:** 1.0  
**تاریخ:** 2026-08-13  
**وضعیت:** ✅ آماده برای استفاده

---

## 📋 فهرست

1. [پیش‌نیازها](#پیش‌نیازها)
2. [نصب و راه‌اندازی](#نصب-و-راه‌اندازی)
3. [روش‌های استفاده](#روش‌های-استفاده)
4. [مثال‌های عملی](#مثال‌های-عملی)
5. [حل مسائل](#حل-مسائل)

---

## ✅ پیش‌نیازها

تمام نیازمندی‌ها **اکنون نصب شده** هستند:

```
✓ Python 3.10+
✓ Tesseract OCR v5.4.0 (نصب شده در C:\Program Files\Tesseract-OCR)
✓ مدل‌های آموزش‌دیده:
  - English (eng.traineddata) 
  - Persian (fas.traineddata)
✓ کتابخانه‌های Python (requirements.txt)
```

### ✅ بررسی نصب

```bash
# برای بررسی نصب صحیح:
cd c:\Users\Hossein\Desktop\ocr
./venv/Scripts/python.exe -c "import pytesseract; print('✓ pytesseract نصب شده')"
```

---

## 🔧 نصب و راه‌اندازی

### مرحله 1: فعال‌کردن محیط مجازی

```bash
# منتقل شو به پوشه پروژه
cd c:\Users\Hossein\Desktop\ocr

# محیط مجازی را فعال کن
# روی Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# یا روی CMD:
.\venv\Scripts\activate.bat
```

### مرحله 2: نصب وابستگی‌ها (اختیاری - اگر هنوز نصب نشده)

```bash
pip install -r requirements.txt
```

### مرحله 3: تصدیق نصب

```bash
# اجرای تست سریع
python -c "from inference import OCREngine; e = OCREngine(); print('✓ سیستم آماده است')"
```

---

## 🎯 روش‌های استفاده

### **روش 1️⃣: رابط کاربری Gradio (ساده‌ترین)**

بهترین برای: **استفاده عمومی و تصویری**

#### اجرا:
```bash
python ui.py
```

#### نتیجه:
- درگاه وب باز می‌شود: `http://localhost:7860`
- کاملاً فارسی است
- بدون نیاز به کد نویسی

#### نحوه استفاده:
1. **برای تصویر:**
   - دکمه "تصویر" را کلیک کن
   - تصویر را بکش و بیفکن
   - "Submit" را کلیک کن
   - متن استخراج‌شده ظاهر می‌شود

2. **برای PDF:**
   - دکمه "فایل PDF" را کلیک کن
   - فایل PDF را انتخاب کن
   - "Submit" را کلیک کن
   - متن تمام صفحات ظاهر می‌شود

---

### **روش 2️⃣: API FastAPI (پیشرفته)**

بهترین برای: **یکپارچه‌سازی با نرم‌افزارهای دیگر**

#### اجرا:
```bash
python api/app.py
```

#### یا با uvicorn:
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

#### نتیجه:
- سرور شروع می‌شود: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

#### استفاده از API:

##### 📸 خواندن متن از تصویر

```bash
# PowerShell:
$imageFile = "C:\path\to\image.png"
Invoke-RestMethod -Uri "http://localhost:8000/ocr/image" `
  -Method Post `
  -Form @{ file = Get-Item $imageFile }

# یا Curl:
curl -X POST "http://localhost:8000/ocr/image" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.png"
```

##### 📄 خواندن متن از PDF

```bash
# PowerShell:
$pdfFile = "C:\path\to\document.pdf"
Invoke-RestMethod -Uri "http://localhost:8000/ocr/pdf" `
  -Method Post `
  -Form @{ file = Get-Item $pdfFile }

# یا Curl:
curl -X POST "http://localhost:8000/ocr/pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

##### 🏥 بررسی وضعیت سرور

```bash
curl http://localhost:8000/health
# Output: {"status":"ok","gpu":false}
```

---

### **روش 3️⃣: کد پایتون (برنامه‌نویسی‌ها)**

بهترین برای: **استفاده در اسکریپت‌ها و برنامه‌های پایتون**

#### مثال ساده:

```python
from inference import OCREngine

# نمونه سازی موتور OCR
ocr = OCREngine()

# خواندن متن از تصویر
result = ocr.read_image('image.png')
print("متن استخراج‌شده:")
print(result['text'])

# یا خواندن از PDF
result = ocr.read('document.pdf')
for page_num, page in enumerate(result['pages'], 1):
    print(f"صفحه {page_num}:")
    print(page['text'])
    print("---")
```

#### مثال پیشرفته (برای تصاویری با زبان‌های مختلف):

```python
from backend.ocr_backend import TesseractOCRBackend
import cv2

# استفاده مستقیم از backend
backend = TesseractOCRBackend(language='fas+eng')

# خواندن تصویر
image = cv2.imread('document.png')

# استخراج متن (زبان خودکار شناسایی می‌شود)
text = backend.recognize(image)
print("متن:", text)
```

---

## 📝 مثال‌های عملی

### ✨ مثال 1: استخراج متن از تصویر فارسی

```python
from inference import OCREngine
from pathlib import Path

ocr = OCREngine()

# فرض کن فایل 'farsi_text.png' داری
result = ocr.read_image('farsi_text.png')

# نتیجه شامل این است:
print("متن:", result['text'])
print("مناطق یافت‌شده:", result['regions'])
print("مسیر اصلی:", result['image_path'])

# ذخیره متن در فایل
with open('extracted_text.txt', 'w', encoding='utf-8') as f:
    f.write(result['text'])
```

### ✨ مثال 2: پردازش چندین تصویر

```python
from inference import OCREngine
from pathlib import Path

ocr = OCREngine()
image_dir = Path('images')

# پردازش تمام تصاویر
for image_file in image_dir.glob('*.png'):
    print(f"در حال پردازش: {image_file.name}")
    result = ocr.read_image(str(image_file))
    
    # ذخیره در فایل متنی
    output_file = image_file.stem + '.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result['text'])
    
    print(f"✓ ذخیره شد: {output_file}\n")
```

### ✨ مثال 3: استخراج متن از PDF فارسی

```python
from inference import OCREngine
from pathlib import Path

ocr = OCREngine()

# خواندن PDF
result = ocr.read('document.pdf')

# نتیجه شامل صفحات است
print(f"تعداد صفحات: {len(result['pages'])}")

for page in result['pages']:
    print(f"\n--- صفحه {page.get('page_num')} ---")
    print(page['text'])
    print(f"اطمینان: {page.get('confidence')}")

# ذخیره تمام متون
all_text = '\n\n---\n\n'.join([p['text'] for p in result['pages']])
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(all_text)
```

### ✨ مثال 4: استفاده از API با Python

```python
import requests
from pathlib import Path

# آدرس سرور API
API_URL = "http://localhost:8000"

# تصویر را بفرست
with open('image.png', 'rb') as f:
    files = {'file': f}
    response = requests.post(f"{API_URL}/ocr/image", files=files)

result = response.json()
print("متن:", result['text'])
```

---

## 🛠️ حل مسائل

### مشکل 1: "Tesseract is not installed"

**حل:**
```bash
# بررسی نصب
"C:\Program Files\Tesseract-OCR\tesseract.exe" --version

# اگر نتیجه ندادید، دوباره نصب کنید:
winget install --id UB-Mannheim.TesseractOCR -e --source winget
```

### مشکل 2: "fas.traineddata not found"

**حل:**
```python
# این خودکار دانلود می‌شود اما اگر مشکل داری:
import os
from pathlib import Path
from backend.ocr_backend import TesseractOCRBackend

# صرفاً ایجاد کن - خودکار دانلود می‌شود
backend = TesseractOCRBackend(language='fas+eng')
```

### مشکل 3: دقت کم OCR

**راه‌حل‌ها:**
1. تصویر‌های با کیفیت بالاتر استفاده کن (حداقل 150 DPI)
2. مطمئن شو تصویر سیاه و سفید (یا خاکستری) است
3. متن‌های کوچک‌تر از 12px را بزرگ کن

```python
import cv2
from inference import OCREngine

img = cv2.imread('image.png')

# بزرگ کردن تصویر برای دقت بیشتر
height, width = img.shape[:2]
if width < 1000:
    scale = 1000 / width
    img = cv2.resize(img, None, fx=scale, fy=scale)

ocr = OCREngine()
result = ocr.read_image(str(img))
print(result['text'])
```

### مشکل 4: محیط مجازی فعال نشده

**حل:**

```bash
# PowerShell:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1

# یا CMD:
.\venv\Scripts\activate.bat
```

---

## 📊 فایل‌های مهم

```
ocr/
├── ui.py                 👈 رابط کاربری (شروع اینجا)
├── api/
│   └── app.py           👈 API سرور
├── inference.py          👈 موتور OCR اصلی
├── backend/
│   └── ocr_backend.py   👈 Tesseract backend
├── preprocessing/        👈 پیش‌پردازش تصاویر
├── postprocessing/       👈 پس‌پردازش متن
└── configs/
    └── config.yaml      👈 تنظیمات
```

---

## 🎓 آموزش سریع

### برای تصویر واحد:
```python
from inference import OCREngine
ocr = OCREngine()
text = ocr.read_image('photo.png')['text']
print(text)
```

### برای PDF:
```python
from inference import OCREngine
ocr = OCREngine()
result = ocr.read('document.pdf')
print(result['pages'][0]['text'])  # صفحه اول
```

### برای API:
```bash
python api/app.py
# سپس به http://localhost:8000/docs بروی و تست کن
```

---

## ✅ نقطه‌های مهم

| نکته | توضیح |
|------|--------|
| 🌐 **زبان** | خودکار فارسی و انگلیسی تشخیص می‌دهد |
| 🚀 **سرعت** | ~200ms برای هر تصویر |
| 📱 **دقت** | بهترین برای متون واضح و خطی |
| ♻️ **استفاده دوباره** | موتور رو یکبار بسازش، بارها استفاده کن |
| 🔄 **concurrent** | برای درخواست‌های همزمان، API بهتر است |

---

## 🎯 خلاصه - سه روش

| روش | دستور | بهترین برای |
|------|-------|-----------|
| **UI** | `python ui.py` | استفاده عمومی، گرافیکی |
| **API** | `python api/app.py` | یکپارچه‌سازی، سرور |
| **کد** | `from inference import OCREngine` | اسکریپت‌ها |

---

**حالا می‌تونی شروع کنی! 🚀**

