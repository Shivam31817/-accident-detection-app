from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect_objects(frame):
    results = model(frame)
    return results[0]
