"""Pre-download models so the first /analyze call is fast."""
from ultralytics import YOLO
import easyocr

print("Downloading YOLOv8n (COCO)…")
YOLO("yolov8n.pt")
print("Downloading EasyOCR English models…")
easyocr.Reader(["en"], gpu=False, verbose=True)
print("Done. Start the server: uvicorn server.main:app --host 0.0.0.0 --port 8000")
