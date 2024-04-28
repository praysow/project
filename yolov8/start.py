from ultralytics import YOLO
import pytesseract
import cv2

# YOLO 모델 로드
model = YOLO('yolov8n.pt')
    # YOLO를 사용하여 프레임에서 객체 탐지
results = model.predict(source='C:/project/yolov8/video/working.mp4', conf=0.7,vid_stride=8,show=True)