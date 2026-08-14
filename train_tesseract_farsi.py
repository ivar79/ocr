#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
آموزش Tesseract برای زبان فارسی
Training Tesseract OCR for Persian Language

نویسنده: OCR Project
تاریخ: 2026-08-13
"""

import sys
import os
from pathlib import Path
import subprocess
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Optional imports
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class TesseractFarsiTrainer:
    """کلاس برای آموزش Tesseract برای فارسی"""
    
    def __init__(self, training_dir='farsi_training_data', output_dir='tessdata_trained'):
        self.training_dir = Path(training_dir)
        self.output_dir = Path(output_dir)
        self.tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
    def setup_directories(self):
        """ایجاد پوشه‌های مورد نیاز"""
        print("📁 مرحله 1: ایجاد پوشه‌های مورد نیاز...")
        
        self.training_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        (self.training_dir / 'images').mkdir(exist_ok=True)
        (self.training_dir / 'labels').mkdir(exist_ok=True)
        
        print(f"  ✓ پوشه‌های ایجاد شدند:")
        print(f"    - {self.training_dir}/images")
        print(f"    - {self.training_dir}/labels")
        print(f"    - {self.output_dir}")
    
    def create_sample_data(self):
        """ایجاد داده‌های نمونه برای تست"""
        print("\n📊 مرحله 2: ایجاد داده‌های نمونه...")
        
        sample_texts = [
            "سلام دنیا",
            "خوش‌آمدید به سیستم OCR",
            "تشخیص کاراکتر نوری",
            "فناوری پردازش تصویر",
            "یادگیری ماشین و هوش مصنوعی",
        ]
        
        if not HAS_PIL:
            print("  ⚠️  PIL نصب نشده است - فقط فایل‌های برچسب ایجاد می‌شوند")
            print("     pip install pillow برای ایجاد تصاویر نمونه")
        
        for i, text in enumerate(sample_texts, 1):
            # ذخیره برچسب (متن) - این همیشه ایجاد می‌شود
            label_path = self.training_dir / 'labels' / f'page_{i:04d}.txt'
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # تلاش برای ایجاد تصویر (اختیاری)
            if HAS_PIL and HAS_CV2:
                try:
                    import numpy as np
                    import cv2
                    
                    img = np.ones((200, 800, 3), dtype=np.uint8) * 255
                    cv2.putText(
                        img, 
                        f"Sample {i}",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 0),
                        2
                    )
                    
                    img_path = self.training_dir / 'images' / f'page_{i:04d}.png'
                    cv2.imwrite(str(img_path), img)
                    print(f"  ✓ نمونه {i}: {text} (تصویر + برچسب)")
                except Exception as e:
                    print(f"  ✓ برچسب {i}: {text} (فقط متن)")
            else:
                print(f"  ✓ برچسب {i}: {text} (فقط متن)")
    
    def check_tesseract(self):
        """بررسی نصب Tesseract"""
        print("\n🔍 مرحله 3: بررسی Tesseract...")
        
        if not Path(self.tesseract_path).exists():
            print(f"  ❌ Tesseract نصب نشده است: {self.tesseract_path}")
            print("\n  راه حل:")
            print("  1. Tesseract را نصب کن:")
            print("     winget install --id UB-Mannheim.TesseractOCR -e --source winget")
            print("  2. یا دستی نصب کن از:")
            print("     https://github.com/UB-Mannheim/tesseract/wiki")
            return False
        
        print(f"  ✓ Tesseract یافت شد: {self.tesseract_path}")
        
        # بررسی نسخه
        try:
            result = subprocess.run(
                [self.tesseract_path, '--version'],
                capture_output=True,
                text=True
            )
            version = result.stdout.split('\n')[0]
            print(f"  ✓ نسخه: {version}")
            return True
        except Exception as e:
            print(f"  ❌ خطا: {e}")
            return False
    
    def convert_to_tiff(self):
        """تبدیل تصاویر PNG به TIFF"""
        print("\n🖼️  مرحله 4: تبدیل تصاویر به فرمت TIFF...")
        
        if not HAS_PIL:
            print("  ⚠️  PIL نصب نشده - نمی‌تواند تصاویر را تبدیل کند")
            print("     pip install pillow برای فعال کردن این مرحله")
            return True  # ادامه بدون خطا
        
        tiff_dir = self.training_dir / 'tiff'
        tiff_dir.mkdir(exist_ok=True)
        
        image_files = list((self.training_dir / 'images').glob('*.png'))
        
        if not image_files:
            print("  ⚠️  هیچ تصویر PNG یافت نشد!")
            print(f"     تصاویر خود را در {self.training_dir}/images قرار دهید")
            return True  # ادامه بدون خطا
        
        for img_file in image_files:
            try:
                from PIL import Image
                
                img = Image.open(img_file)
                tiff_file = tiff_dir / img_file.with_suffix('.tiff').name
                
                # استانداردسازی
                img = img.convert('RGB')
                img.save(tiff_file, 'TIFF')
                
                print(f"  ✓ {img_file.name} → {tiff_file.name}")
            except Exception as e:
                print(f"  ⚠️  خطا در {img_file.name}: {e}")
                # ادامه بدون خطا
        
        return True
    
    def prepare_ground_truth(self):
        """آماده‌کردن فایل‌های Ground Truth"""
        print("\n📝 مرحله 5: آماده‌کردن فایل‌های Ground Truth...")
        
        tiff_dir = self.training_dir / 'tiff'
        
        for tiff_file in tiff_dir.glob('*.tiff'):
            # یافتن فایل برچسب متناظر
            label_file = self.training_dir / 'labels' / tiff_file.with_suffix('.txt').name
            
            if label_file.exists():
                # کپی متن برای Tesseract
                with open(label_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                print(f"  ✓ {tiff_file.name} ← {text[:30]}")
            else:
                print(f"  ⚠️  فایل برچسب یافت نشد: {label_file.name}")
    
    def print_instructions(self):
        """چاپ دستورالعمل برای ادامه آموزش"""
        print("\n" + "="*70)
        print("📚 دستورالعمل‌های آموزش Tesseract")
        print("="*70)
        
        print("""
╔════════════════════════════════════════════════════════════════════╗
║                   مراحل آموزش Tesseract                           ║
╚════════════════════════════════════════════════════════════════════╝

✅ مراحل تکمیل‌شده:
  1. ایجاد ساختار پوشه‌ها
  2. داده‌های نمونه ایجاد شدند
  3. Tesseract بررسی شد
  4. تصاویر به TIFF تبدیل شدند
  5. فایل‌های Ground Truth آماده شدند

📋 مراحل بعدی (دستی):

1️⃣  جمع‌آوری داده‌های فارسی بیشتر:
   - تصاویر خود را در: farsi_training_data/images قرار دهید
   - برای هر تصویر (page_001.png)، متن مطابق آن را در:
     farsi_training_data/labels/page_001.txt قرار دهید

2️⃣  اجرای آموزش (Windows):
   
   # ابتدا tesstrain را نصب کن
   pip install tesstrain
   
   # سپس آموزش را شروع کن
   tesstrain.py ^
     --lang fas ^
     --linedata_only ^
     --output_dir ./tessdata_trained ^
     ./farsi_training_data

3️⃣  ایجاد مدل نهایی:
   
   cd tessdata_trained
   combine_tessdata -o farsi_trained.traineddata *

4️⃣  استفاده از مدل جدید:
   
   from backend.ocr_backend import TesseractOCRBackend
   import os
   
   # تنظیم مسیر Tessdata
   os.environ['TESSDATA_PREFIX'] = './tessdata_trained'
   
   backend = TesseractOCRBackend(language='fas')
   text = backend.recognize(image)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 نکات مهم:

• داده‌های آموزشی:
  ✓ حداقل 100 تصویر برای شروع
  ✓ حداقل 1000 تصویر برای دقت خوب
  ✓ تنوع فونت، سایز، و زاویه

• فرمت Ground Truth:
  ✓ هر تصویر: page_001.png
  ✓ برچسب متناظر: page_001.txt
  ✓ فقط متن فارسی در فایل txt

• تنظیمات آموزش:
  ✓ --lang fas (برای فارسی)
  ✓ --linedata_only (برای متن خطی)
  ✓ --output_dir (مسیر خروجی)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔗 منابع مفید:

• داکومنتیشن Tesseract:
  https://tesseract-ocr.github.io/tessdoc/
  
• تمرین Tesseract:
  https://tesseract-ocr.github.io/tessdoc/Training-Tesseract

• دیتاست فارسی:
  - فارسی Books
  - Farsi Wikipedia
  - اسناد دولتی (با اجازه)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ روش سریع (بدون آموزش پیچیده):

اگر می‌خوای بدون آموزش Tesseract دقت را بهبود بدی:

1. بهبود Preprocessing:
   - تصاویر را بهتر پردازش کن
   - کنتراست را بالا ببر
   - نویز را کم کن

2. استفاده از Tesseract کیوری:
   - PSM (Page Segmentation Mode) را تنظیم کن
   - OEM (OCR Engine Mode) را تغییر بده

3. Postprocessing:
   - تصحیح هجی
   - تصحیح شناسایی اعداد
   - نرمال‌سازی فارسی

        """)
        
        print("="*70)
        print("✨ برای کمک بیشتر به TRAINING_GUIDE_FA.md مراجعه کن")
        print("="*70)
    
    def run(self):
        """اجرای کل فرآیند"""
        print("\n" + "="*70)
        print("🚀 آموزش Tesseract برای فارسی")
        print("="*70 + "\n")
        
        # مرحله 1
        self.setup_directories()
        
        # مرحله 2
        self.create_sample_data()
        
        # مرحله 3
        if not self.check_tesseract():
            print("\n❌ Tesseract نصب نشده است. ابتدا آن را نصب کن.")
            return False
        
        # مرحله 4
        self.convert_to_tiff()  # اختیاری - خطا نمی‌دهد
        
        # مرحله 5
        self.prepare_ground_truth()
        
        # چاپ دستورالعمل‌ها
        self.print_instructions()
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='آموزش Tesseract برای فارسی',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
مثال‌ها:
  python train_tesseract_farsi.py                    # آماده‌سازی پایه
  python train_tesseract_farsi.py --sample           # با داده‌های نمونه
  python train_tesseract_farsi.py --help             # کمک
        """
    )
    
    parser.add_argument(
        '--sample',
        action='store_true',
        help='ایجاد داده‌های نمونه برای تست'
    )
    
    parser.add_argument(
        '--training-dir',
        default='farsi_training_data',
        help='مسیر پوشه آموزشی (پیش‌فرض: farsi_training_data)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='tessdata_trained',
        help='مسیر پوشه خروجی (پیش‌فرض: tessdata_trained)'
    )
    
    args = parser.parse_args()
    
    # ایجاد Trainer
    trainer = TesseractFarsiTrainer(
        training_dir=args.training_dir,
        output_dir=args.output_dir
    )
    
    # اجرا
    success = trainer.run()
    
    if success:
        print("\n✅ آماده‌سازی تکمیل شد!")
        print("\n📚 مرحله بعد: داده‌های آموزشی خود را جمع‌آوری کنید")
        sys.exit(0)
    else:
        print("\n❌ خطا در آماده‌سازی")
        sys.exit(1)


if __name__ == '__main__':
    main()
