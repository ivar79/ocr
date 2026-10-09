# Persian OCR - Farsi Text Recognition

A deep learning-based OCR system for Persian (Farsi) text recognition with support for English as well. Uses CRNN (Convolutional Recurrent Neural Network) for recognition and Tesseract as a fallback backend.

## Features

- **Dual Backend Support**: CRNN deep learning model and Tesseract OCR
- **Persian & English**: Supports both Persian (Farsi) and English text
- **Adaptive Preprocessing**: Smart image preprocessing for different quality levels
- **Post-processing**: Persian text normalization and character correction
- **GPU Training**: Train on Google Colab with GPU acceleration
- **Batch Processing**: Process multiple images or PDFs

## Current Status

- ✅ Tesseract backend functional (basic accuracy)
- ⚠️ CRNN model needs retraining (current CER: 0.8558 - poor accuracy)
- 🔄 Training pipeline ready for improvement

## Installation

### Prerequisites

- Python 3.8+
- Tesseract OCR 5.x
- CUDA (optional, for GPU training)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/ivar79/ocr.git
cd ocr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Install Tesseract

**Windows:**
```bash
# Using winget
winget install UB-Mannheim.Tesseract

# Or download from: https://github.com/UB-Mannheim/tesseract/wiki
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-fas
```

**macOS:**
```bash
brew install tesseract
brew install tesseract-lang
```

## Quick Start

### Basic Usage

```python
from inference import OCREngine

# Initialize engine
engine = OCREngine()

# Read image
result = engine.read_image("path/to/image.png")
print(result["text"])
print(f"Confidence: {result['confidence']:.2f}")

# Read PDF
result = engine.read("path/to/document.pdf")
print(result["text"])
```

### Command Line

```bash
# Run inference
python inference.py --image path/to/image.png

# Benchmark OCR performance
python benchmark_ocr.py --backend tesseract --images test_samples
```

## Training

### Local Training (CPU)

```bash
# Train recognition model
python train.py --mode recognition --config configs/config.yaml

# Train detection model
python train.py --mode detection --config configs/config.yaml
```

### Training on Google Colab (GPU)

For faster training with GPU, use the provided Colab notebook:

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Enable GPU: Runtime → Change runtime type → GPU
4. Copy the cells from `notebooks/train_on_colab.ipynb`
5. Run cells in order

**Or clone and run the notebook directly:**
```bash
# In Colab
!git clone https://github.com/ivar79/ocr.git
%cd ocr
# Then open and run notebooks/train_on_colab.ipynb
```

### Training Configuration

Edit `configs/config.yaml` to adjust training parameters:

```yaml
training:
  learning_rate: 0.003
  batch_size: 64          # Increase for GPU, decrease for CPU
  epochs: 200             # Number of training epochs
  patience: 20            # Early stopping patience
  min_epochs: 30          # Minimum epochs before early stopping

recognition:
  hidden_size: 512        # Model capacity (128, 256, 512)
  cnn_scale: 1.0          # CNN scale (0.5 for speed, 1.0 for accuracy)
```

## Configuration

### Backend Selection

Edit `configs/config.yaml`:

```yaml
ocr:
  backend: "tesseract"    # Options: tesseract, crnn
  language: "fas+eng"     # Persian + English
  enable_retry: true
  confidence_threshold: 0.5
```

### Preprocessing

```yaml
preprocessing:
  target_size: [640, 640]
  upscale_factor: 2
  clahe_clip_limit: 3.0
  denoise_strength: 75
  mode: "auto"            # auto, fast, full
```

## Project Structure

```
ocr/
├── configs/
│   └── config.yaml         # Configuration file
├── backend/
│   ├── ocr_backend.py      # Tesseract backend
│   └── crnn_backend.py     # CRNN backend
├── preprocessing/
│   ├── image_processing.py # Image preprocessing
│   └── augmentation.py     # Data augmentation
├── postprocessing/
│   └── text_formatter.py   # Persian text normalization
├── models/
│   ├── detection/          # Text detection models
│   ├── recognition/        # Text recognition models
│   └── pretrained/         # Trained model weights
├── utils/
│   ├── dataset.py          # Dataset utilities
│   ├── metrics.py          # OCR metrics (CER, WER)
│   └── tesseract_setup.py  # Tesseract configuration
├── data/
│   ├── train/              # Training data
│   ├── val/                # Validation data
│   └── test/               # Test data
├── notebooks/
│   └── train_on_colab.ipynb # Colab training notebook
├── train.py                # Training script
├── inference.py            # Inference engine
├── benchmark_ocr.py        # Benchmark script
└── requirements.txt        # Python dependencies
```

## Dataset

### Training Data Format

Place training images in `data/train/images/` and labels in `data/train/labels.txt`:

```
train_00000.png	سلام دنیا
train_00001.png	این یک تست است
```

### Recommended Datasets

For better Persian OCR accuracy, consider using:

1. **IUST Persian Handwritten Dataset**
   - Handwritten Persian text
   - Good for CRNN training

2. **IIIT-HW-POST**
   - Multi-language dataset including Persian
   - High quality

3. **Synthetic Data**
   - Generate with Persian fonts
   - See `scripts/generate_synthetic_data.py`

## Performance

### Current Metrics

- **Tesseract**: Functional for basic text (CER ~0.5-0.7)
- **CRNN**: Needs retraining (CER ~0.86 - poor)

### Target Metrics

- **CER (Character Error Rate)**: < 0.1
- **WER (Word Error Rate)**: < 0.2
- **Processing Time**: < 200ms per image

## Troubleshooting

### Tesseract Not Found

```bash
# Check installation
tesseract --version

# Check language data
tesseract --list-langs

# Download Persian language data
# Windows: Download fas.traineddata to tessdata folder
# Linux: sudo apt-get install tesseract-ocr-fas
```

### CUDA Out of Memory

Reduce batch size in `configs/config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 64
```

### Poor Recognition Accuracy

1. Check image quality and contrast
2. Try different preprocessing modes (fast/full)
3. Ensure Persian language data is installed
4. Retrain CRNN model with better data

## Contributing

Contributions are welcome! Areas for improvement:

- Better Persian datasets
- Improved preprocessing
- Additional language support
- Performance optimization
- UI/UX improvements

## License

MIT License

## Acknowledgments

- Tesseract OCR engine
- PyTorch framework
- Persian language community

## Contact

For issues and questions, please open an issue on GitHub.
