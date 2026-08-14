from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os
import torch
import sys
from pathlib import Path

# اضافه کردن مسیر روت به path برای دسترسی به inference
sys.path.append(str(Path(__file__).parent.parent))
from inference import OCREngine

app = FastAPI(title="Persian OCR API", version="1.0.0")
ocr = OCREngine()

@app.post("/ocr/image")
async def ocr_image(file: UploadFile = File(...)):
    """خواندن متن از تصویر"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = ocr.read_image(tmp_path)
        return JSONResponse(content=result)
    finally:
        os.unlink(tmp_path)

@app.post("/ocr/pdf")
async def ocr_pdf(file: UploadFile = File(...)):
    """خواندن متن از PDF"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = ocr.read(tmp_path)
        return JSONResponse(content=result)
    finally:
        os.unlink(tmp_path)

@app.get("/health")
async def health():
    return {"status": "ok", "gpu": torch.cuda.is_available()}
