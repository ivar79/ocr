import torch
import torch.nn as nn
import time
import argparse
from pathlib import Path
from models.detection.craft import CRAFT
from models.recognition.crnn import CRNN

def benchmark_model(model, dummy_input, name="Model", num_iters=100):
    print(f"Benchmarking {name}...")
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
            
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(dummy_input)
    end = time.time()
    
    avg_time = (end - start) / num_iters * 1000
    print(f"{name} Avg Inference Time: {avg_time:.2f} ms")
    return avg_time

def optimize_onnx():
    print("\n--- Exporting to ONNX ---")
    out_dir = Path("models/pretrained")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # CRAFT
    craft = CRAFT(pretrained=False).eval()
    dummy_craft = torch.randn(1, 3, 640, 640)
    torch.onnx.export(
        craft, dummy_craft, out_dir / "craft.onnx",
        input_names=["image"], output_names=["region", "affinity"],
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}}
    )
    print("Exported CRAFT to ONNX.")
    
    # CRNN
    crnn = CRNN(32, 1, 100).eval()
    dummy_crnn = torch.randn(1, 1, 32, 128)
    torch.onnx.export(
        crnn, dummy_crnn, out_dir / "crnn.onnx",
        input_names=["image"], output_names=["text"],
        dynamic_axes={"image": {0: "batch", 3: "width"}}
    )
    print("Exported CRNN to ONNX.")

def optimize_quantize():
    print("\n--- Dynamic Quantization ---")
    crnn = CRNN(32, 1, 100).eval()
    dummy_input = torch.randn(1, 1, 32, 128)
    
    quantized_crnn = torch.quantization.quantize_dynamic(
        crnn, {nn.Linear, nn.LSTM}, dtype=torch.qint8
    )
    
    benchmark_model(crnn, dummy_input, "PyTorch Original CRNN")
    benchmark_model(quantized_crnn, dummy_input, "Quantized INT8 CRNN")

def optimize_torchscript():
    print("\n--- TorchScript Export ---")
    out_dir = Path("models/pretrained")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    craft = CRAFT(pretrained=False).eval()
    # Need strict=False for modules with dictionary outputs or complex control flow sometimes
    scripted_craft = torch.jit.script(craft)
    scripted_craft.save(str(out_dir / "craft_scripted.pt"))
    print("Exported CRAFT to TorchScript.")
    
    crnn = CRNN(32, 1, 100).eval()
    scripted_crnn = torch.jit.script(crnn)
    scripted_crnn.save(str(out_dir / "crnn_scripted.pt"))
    print("Exported CRNN to TorchScript.")

def run_benchmark():
    print("\n--- Running Full Benchmark ---")
    optimize_quantize()
    # In a full implementation, onnxruntime would be loaded and benchmarked here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["onnx", "quantize", "torchscript", "benchmark"], default="benchmark")
    args = parser.parse_args()
    
    if args.mode == "onnx":
        optimize_onnx()
    elif args.mode == "quantize":
        optimize_quantize()
    elif args.mode == "torchscript":
        optimize_torchscript()
    else:
        run_benchmark()
