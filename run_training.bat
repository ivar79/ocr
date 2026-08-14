@echo off
REM آموزش Tesseract برای فارسی - فایل اجرایی Windows

cd /d c:\Users\Hossein\Desktop\ocr

echo ╔════════════════════════════════════════════╗
echo ║   آموزش Tesseract برای فارسی              ║
echo ║   Training Setup Script                    ║
echo ╚════════════════════════════════════════════╝
echo.

echo فعال‌کردن محیط مجازی...
call venv\Scripts\activate.bat

echo.
echo اجرای اسکریپت آموزش...
python train_tesseract_farsi.py

pause
