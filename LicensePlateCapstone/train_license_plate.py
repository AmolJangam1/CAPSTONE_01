from pathlib import Path
from ultralytics import YOLO

def train_model():
    # Automatically resolve path relative to this script's location
    base_dir = Path(__file__).resolve().parent
    dataset_yaml = base_dir / "dataset" / "data.yaml"

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Could not find dataset configuration at: {dataset_yaml}")

    # Load pretrained YOLOv8 Nano model
    model = YOLO("yolov8n.pt")

    print("Starting YOLOv8 training on license plate dataset...")
    model.train(
        data=str(dataset_yaml),  # Passes absolute path to dataset/data.yaml
        epochs=30,
        imgsz=320,               # Matches mobile deployment resolution
        batch=16,
        workers=0,               # Set to 0 on Windows to avoid multi-processing locks
        project=str(base_dir / "runs" / "detect"),
        name="license_plate_model",
        exist_ok=True
    )

    print("\nTraining complete!")

if __name__ == "__main__":
    train_model()