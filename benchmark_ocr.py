"""Benchmark the OCR pipeline on a folder of images.

For every image in ``--images`` the script measures:
  * end-to-end wall time,
  * number of Tesseract subprocess calls (the old pipeline needed up to 8
    per image; the optimized one needs at most 3),
  * recognized text, and
  * CER / WER against a ground-truth file when one exists.

Usage:
    python benchmark_ocr.py --backend tesseract [--images test_samples]
    python benchmark_ocr.py --backend crnn --images test_samples

Ground truth files live in ``--ground-truth`` as ``<image-basename>.txt``.
"""

import argparse
import time
from pathlib import Path

import cv2
import yaml

import pytesseract as _pt

from utils.tesseract_setup import configure_tesseract
configure_tesseract()

from backend.ocr_backend import OCRBackendFactory
from utils.metrics import character_error_rate, word_error_rate

# ---------------------------------------------------------------------------
# Count actual Tesseract invocations (each runs a separate subprocess)
# ---------------------------------------------------------------------------
_call_counts = {"image_to_data": 0, "image_to_string": 0}


def _wrap(fn, key):
    def wrapper(*args, **kwargs):
        _call_counts[key] += 1
        return fn(*args, **kwargs)
    return wrapper


_pt.image_to_data = _wrap(_pt.image_to_data, "image_to_data")
_pt.image_to_string = _wrap(_pt.image_to_string, "image_to_string")


def load_config():
    try:
        with open("configs/config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def build_backend(name: str):
    config = load_config()
    ocr_cfg = config.get("ocr", {})
    kwargs = {
        "enable_retry": ocr_cfg.get("enable_retry", True),
        "confidence_threshold": ocr_cfg.get("confidence_threshold", 0.5),
        "max_retries": ocr_cfg.get("max_retries", 2),
    }
    if name == "tesseract":
        kwargs["preprocessing_config"] = config.get("preprocessing", {})
    else:
        kwargs["weights"] = (
            ocr_cfg.get("crnn_weights")
            or config.get("recognition", {}).get("weights",
                                                 "models/pretrained/crnn_weights.pth"))
    return OCRBackendFactory.create(backend=name, language=ocr_cfg.get("language", "fas+eng"),
                                    **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="test_samples")
    parser.add_argument("--ground-truth", default="test_samples/ground_truth")
    parser.add_argument("--backend", choices=["tesseract", "crnn"], default="tesseract")
    parser.add_argument("--out", default=None, help="Write results to this file")
    args = parser.parse_args()

    images_dir = Path(args.images)
    gt_dir = Path(args.ground_truth)
    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))
    if not images:
        print(f"No images found in {images_dir}")
        return

    backend = build_backend(args.backend)
    print(f"Backend: {args.backend} ({type(backend).__name__})")
    print(f"Images:  {len(images)} from {images_dir}")
    print()

    rows = []
    totals = {"time": 0.0, "calls": 0, "cer": 0.0, "wer": 0.0}
    n_gt = 0

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"SKIP {img_path.name}: unreadable")
            continue
        _call_counts["image_to_data"] = 0
        _call_counts["image_to_string"] = 0

        t0 = time.perf_counter()
        details = backend.recognize_with_details(image)
        elapsed = time.perf_counter() - t0
        calls = _call_counts["image_to_data"] + _call_counts["image_to_string"]
        text = details.get("text", "")
        confidence = details.get("confidence", 0.0)

        gt_file = gt_dir / f"{img_path.name}.txt"
        cer = wer = None
        if gt_file.exists():
            target = gt_file.read_text(encoding="utf-8").strip()
            cer = character_error_rate(text, target)
            wer = word_error_rate(text, target)
            n_gt += 1
            totals["cer"] += cer
            totals["wer"] += wer

        totals["time"] += elapsed
        totals["calls"] += calls
        rows.append((img_path.name, elapsed, calls, confidence, cer, wer, text))

    # Report
    lines = []
    lines.append(f"BACKEND: {args.backend}")
    lines.append(f"{'image':46s} {'time(s)':>8s} {'calls':>5s} {'conf':>6s} "
                 f"{'CER':>7s} {'WER':>7s}")
    lines.append("-" * 90)
    for name, elapsed, calls, conf, cer, wer, _ in rows:
        cer_s = f"{cer:.3f}" if cer is not None else "-"
        wer_s = f"{wer:.3f}" if wer is not None else "-"
        lines.append(f"{name[:46]:46s} {elapsed:8.2f} {calls:5d} {conf:6.2f} "
                     f"{cer_s:>7s} {wer_s:>7s}")
    lines.append("-" * 90)
    avg_time = totals["time"] / max(len(rows), 1)
    avg_calls = totals["calls"] / max(len(rows), 1)
    lines.append(f"AVG time: {avg_time:.2f}s   AVG tesseract calls: {avg_calls:.1f}")
    if n_gt:
        lines.append(f"AVG CER: {totals['cer']/n_gt:.3f}   AVG WER: {totals['wer']/n_gt:.3f} "
                     f"(over {n_gt} images with ground truth)")
    lines.append("")
    for name, _, _, _, _, _, text in rows:
        lines.append(f"--- {name} ---")
        lines.append(text)
        lines.append("")

    report = "\n".join(lines)
    print(report)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
