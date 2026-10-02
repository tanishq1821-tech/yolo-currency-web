from ultralytics import YOLO
import cv2

# 1. Load your trained model weights
model_path = r"runs\detect\inr_mendeley_model-9\weights\best.pt"
model = YOLO(model_path)

# 2. Path to your test image
image_path = "500_test.jpg"

# 3. Run inference
results = model.predict(source=image_path, conf=0.25, save=True, show_labels=True, show_conf=True)

# 4. Print detected objects & confidence scores in terminal
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0]) * 100
        print(f"Detected: {class_name} | Confidence: {confidence:.2f}%")

print(f"\nResult image saved to: {results[0].save_dir}")