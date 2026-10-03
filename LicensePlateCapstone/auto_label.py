import os
import shutil
import cv2
from ultralytics import YOLO

def auto_label_dataset():
    # 1. Load a pre-trained model that can detect license plates or vehicles
    # Using open-source YOLOv8 license plate weights or general detector
    model = YOLO("yolov8n.pt")  

    raw_dir = "raw_images"
    dataset_dir = "dataset"
    
    # Create YOLO directory layout
    for split in ["train", "val"]:
        os.makedirs(f"{dataset_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{dataset_dir}/{split}/labels", exist_ok=True)

    image_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} unlabeled images.")

    # Split 80% for training, 20% for validation
    split_index = int(len(image_files) * 0.8)
    
    for idx, img_name in enumerate(image_files):
        split = "train" if idx < split_index else "val"
        src_img_path = os.path.join(raw_dir, img_name)
        
        # Run inference to detect objects
        results = model.predict(src_img_path, conf=0.25, verbose=False)[0]
        
        # Copy image to target directory
        dst_img_path = os.path.join(dataset_dir, split, "images", img_name)
        shutil.copy(src_img_path, dst_img_path)
        
        # Write YOLO label format (.txt)
        label_file = os.path.splitext(img_name)[0] + ".txt"
        dst_label_path = os.path.join(dataset_dir, split, "labels", label_file)
        
        img_h, img_w = results.orig_shape
        
        with open(dst_label_path, "w") as f:
            for box in results.boxes:
                # Class 0: license_plate
                cls = 0  
                # Normalize bounding box coordinates (0.0 to 1.0)
                x_center, y_center, w, h = box.xywhn[0].tolist()
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print("\nDataset successfully organized and auto-labeled in 'dataset/'!")

if __name__ == "__main__":
    auto_label_dataset()