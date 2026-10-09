from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import torch
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from inference import OCREngine

app = FastAPI(title="Persian OCR API", version="1.0.0")

# Add CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR engine
ocr = OCREngine()


@app.post("/ocr/image")
async def ocr_image(file: UploadFile = File(...)):
    """Read text from an image file."""
    tmp_path = None
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process image
        result = ocr.read_image(tmp_path)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Always cleanup temp file
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.post("/ocr/pdf")
async def ocr_pdf(file: UploadFile = File(...)):
    """Read text from a PDF file."""
    tmp_path = None
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process PDF
        result = ocr.read(tmp_path)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Always cleanup temp file
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "gpu": torch.cuda.is_available()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
