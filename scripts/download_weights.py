import os
import urllib.request
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm is not installed. Progress bar will not be shown.")
    tqdm = None

try:
    import gdown
except ImportError:
    gdown = None

class DownloadProgressBar:
    def __init__(self, desc="Downloading"):
        self.pbar = None
        self.desc = desc

    def update_to(self, b=1, bsize=1, tsize=None):
        if tqdm is not None:
            if self.pbar is None:
                self.pbar = tqdm(total=tsize, desc=self.desc, unit='B', unit_scale=True)
            self.pbar.update(b * bsize - self.pbar.n)
        else:
            if tsize is not None and b % 10 == 0:
                print(f"{self.desc}: {b*bsize/1024/1024:.2f} MB / {tsize/1024/1024:.2f} MB")

def _looks_like_html(path: Path) -> bool:
    try:
        with path.open('rb') as f:
            head = f.read(256)
        text = head.lower()
        return b'<!doctype html' in text or b'<html' in text or b'cannot retrieve' in text
    except Exception:
        return False


def _is_valid_pytorch_checkpoint(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    if _looks_like_html(path):
        return False
    if torch is None:
        return True
    try:
        torch.load(path, map_location='cpu', weights_only=False)
        return True
    except Exception:
        return False


def download_file(url, dest_path, desc="Downloading"):
    dest_path = Path(dest_path)
    if dest_path.exists() and _is_valid_pytorch_checkpoint(dest_path):
        print(f"File {dest_path.name} already exists and looks valid. Skipping download.")
        return True
    if dest_path.exists():
        print(f"Removing invalid cached file: {dest_path}")
        dest_path.unlink()

    print(f"Downloading {dest_path.name}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        progress = DownloadProgressBar(desc=desc)

        if "drive.google.com" in url and gdown is not None:
            gdown.download(url, str(dest_path), quiet=False, fuzzy=True)
        else:
            urllib.request.urlretrieve(url, str(dest_path), reporthook=progress.update_to)

        if progress.pbar:
            progress.pbar.close()

        if not _is_valid_pytorch_checkpoint(dest_path):
            print(f"Downloaded file for {dest_path.name} is not a valid PyTorch checkpoint.")
            return False

        print(f"Successfully downloaded {dest_path.name}")
        return True
    except Exception as e:
        print(f"Error downloading {dest_path.name}: {e}")
        return False

if __name__ == "__main__":
    weights_dir = Path("models/pretrained")

    craft_urls = [
        "https://drive.google.com/uc?export=download&id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ",
        "https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ",
    ]
    crnn_urls = [
        "https://www.dropbox.com/s/8p6p8w9x9r5x5m6/crnn.pth?dl=1",
        "https://github.com/meijieru/crnn.pytorch/raw/master/data/crnn.pth",
    ]

    craft_ok = False
    for url in craft_urls:
        if download_file(url, weights_dir / "craft_weights.pth", "CRAFT Weights"):
            craft_ok = True
            break

    crnn_ok = False
    for url in crnn_urls:
        if download_file(url, weights_dir / "crnn_weights.pth", "CRNN Weights"):
            crnn_ok = True
            break

    if craft_ok and crnn_ok:
        print("\nAll weights downloaded successfully.")
    else:
        print("\nOne or more weights could not be downloaded. The app will run with fallback dummy models until valid weights are added.")
