import os
from pathlib import Path
from ultralytics import YOLO
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

def run_export_and_quantization():
    base_dir = Path(__file__).resolve().parent
    pt_model_path = base_dir / "runs" / "detect" / "license_plate_model" / "weights" / "best.pt"

    if not pt_model_path.exists():
        raise FileNotFoundError(f"PyTorch weights not found at: {pt_model_path}")

    print("--- STEP 1: EXPORT PYTORCH TO FP32 ONNX ---")
    model = YOLO(str(pt_model_path))
    
    # Export to standard ONNX FP32
    fp32_onnx_path = model.export(
        format="onnx",
        imgsz=320,        # Match input dimensions for mobile frame processing
        opset=13,         # Compatible with ONNX Runtime Mobile C++ engine
        simplify=True     # Simplify ONNX execution graph
    )
    
    fp32_onnx_path = Path(fp32_onnx_path)
    print(f"FP32 ONNX exported successfully: {fp32_onnx_path}")

    print("\n--- STEP 2: DYNAMIC INT8 QUANTIZATION ---")
    int8_onnx_path = fp32_onnx_path.parent / "best_int8.onnx"

    # Quantize FP32 ONNX model to 8-bit integers
    quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        weight_type=QuantType.QUInt8
    )
    print(f"INT8 Quantized ONNX saved at: {int8_onnx_path}")

    # Calculate Compression Ratios for Capstone Report
    fp32_size = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_onnx_path) / (1024 * 1024)
    reduction = (1 - (int8_size / fp32_size)) * 100

    print("\n================ CAPSTONE METRICS ================")
    print(f"Original FP32 ONNX Size: {fp32_size:.2f} MB")
    print(f"Quantized INT8 ONNX Size: {int8_size:.2f} MB")
    print(f"File Size Compression Ratio: {reduction:.2f}% reduction")
    print("==================================================")

if __name__ == "__main__":
    run_export_and_quantization()