import os
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained YOLOv8 model
# Use relative path instead of absolute C:\ paths
MODEL_PATH = os.path.join(
    "runs", "detect", "inr_mendeley_model-9", "weights", "best.pt"
)
model = YOLO(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run YOLO inference
        results = model.predict(
            source=filepath,
            conf=0.35,  # Confidence threshold
            iou=0.45    # NMS IoU threshold to prevent duplicate boxes
        )

        # Draw bounding boxes and labels onto the image
        annotated_frame = results[0].plot()
        
        # Save output image with detections
        output_filename = f"detected_{filename}"
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_filepath, annotated_frame)

        # Collect detected object labels and confidences
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0]) * 100
            detections.append({
                'class': class_name,
                'confidence': f"{confidence:.2f}%"
            })

        return jsonify({
            'original_image': f"/static/uploads/{filename}",
            'detected_image': f"/static/uploads/{output_filename}",
            'detections': detections
        })

    return jsonify({'error': 'Invalid file type. Please upload a JPG or PNG.'}), 400

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port)