from ultralytics import YOLO

def main():
    # Load lightweight nano model pre-trained on COCO
    model = YOLO("yolov8n.pt") 

    print("--- Starting CPU Model Training ---")
    
    # Train model on CPU
    results = model.train(
        data=r"C:\Users\tanis\OneDrive\Documents\dataset\IndianBankNotes\data.yaml",
        epochs=30,
        imgsz=416,
        batch=8,
        device="cpu",
        workers=4,
        name="inr_mendeley_model"
    )
    
    print("\nTraining complete! Weights saved to: runs/detect/inr_mendeley_model/weights/best.pt")

if __name__ == "__main__":
    main()