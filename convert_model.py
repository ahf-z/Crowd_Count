import torch
from ultralytics import YOLO

# Load your YOLOv8n PyTorch model
model = YOLO('yolov8n.pt') # Ensure yolov8n.pt is in the same directory as this script

# Define input image size (YOLOv8n typically uses 640x640)
imgsz = 640

print("Exporting to TFLite (Float32)...")
# Export to TFLite (float32)
# The `half=False` is important for float32 output
# `int8` quantization can be explored later for smaller size but might impact accuracy.
model.export(format='tflite', imgsz=imgsz, half=False, int8=False)
print(f"TFLite model exported as yolov8n_float32.tflite")

print("Exporting to Core ML (for iOS)...")
# Export to Core ML
# `nms=True` includes Non-Maximum Suppression directly in the Core ML model, simplifying post-processing in Swift/Objective-C.
model.export(format='coreml', imgsz=imgsz, nms=True)
print(f"Core ML model exported as yolov8n.mlmodel (or yolov8n.mlpackage)")

print("Conversion complete!")