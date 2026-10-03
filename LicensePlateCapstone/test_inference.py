from ultralytics import YOLO
import cv2

# Load fine-tuned PyTorch weights
model = YOLO("runs/detect/license_plate_model/weights/best.pt")

# Perform prediction (replace with path to your test car image)
results = model.predict(source="test_car.jpg", conf=0.5)

# Save annotated result to disk so you can view it directly in VS Code
annotated_frame = results[0].plot()
cv2.imwrite("test_output.jpg", annotated_frame)

print("Inference completed! Open 'test_output.jpg' in VS Code to view bounding box.")