import json
import os
import cv2
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- INITIALIZE FIREBASE ADMIN SDK ---
db = None

if os.path.exists("serviceAccountKey.json"):
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
elif os.environ.get("FIREBASE_CREDS_JSON"):
    creds_raw = os.environ.get("FIREBASE_CREDS_JSON", "").strip()
    if creds_raw:
        try:
            service_account_info = json.loads(creds_raw)
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
        except json.JSONDecodeError as e:
            print(f"Error parsing FIREBASE_CREDS_JSON: {e}")
    else:
        print("FIREBASE_CREDS_JSON variable is empty.")
else:
    print("Warning: No Firebase credentials found.")

UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(
    "runs", "detect", "inr_mendeley_model-9", "weights", "best.pt"
)
model = YOLO(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is not None:
            img_resized = cv2.resize(img, (640, 640))
            cv2.imwrite(filepath, img_resized)

        results = model.predict(
            source=filepath, imgsz=640, conf=0.35, iou=0.45
        )

        annotated_frame = results[0].plot()
        output_filename = f"detected_{filename}"
        output_filepath = os.path.join(
            app.config["UPLOAD_FOLDER"], output_filename
        )
        cv2.imwrite(output_filepath, annotated_frame)

        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0]) * 100
            detections.append(
                {"class": class_name, "confidence": f"{confidence:.2f}%"}
            )

        # Safely attempt database write only if Firebase initialized successfully
        if db is not None:
            try:
                db.collection("currency_detections").add({
                    "filename": filename,
                    "detections": detections,
                    "timestamp": firestore.SERVER_TIMESTAMP,
                })
            except Exception as e:
                print(f"Firebase logging error: {e}")

        return jsonify({"status": "success", "detections": detections})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)