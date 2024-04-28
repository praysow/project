from ultralytics import YOLO
import pytesseract
import cv2

# YOLO 모델 로드
model = YOLO('yolov8n.pt')

# 비디오 파일 로드
cap = cv2.VideoCapture('C:/project/yolov8/video/working.mp4')

# 프레임별로 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO를 사용하여 프레임에서 객체 탐지
    results = model.predict(frame, conf=0.7)
    
    # 탐지된 객체들을 순회하면서
    for det in results:
        # 객체가 탐지된 경우에만 처리
        box = det.boxes
        if det.probs is not None:
            for i in range(len(box.xyxy)):
                x1, y1, x2, y2 = box.xyxy[i].tolist()  # 각 좌표를 추출하여 변수에 저장
                x_center1 = (round(x1)+round(x2))/2
                y_center1 = (round(y1)+round(y2))/2
                if x_center1 < 720/2:
                    x_center2='left'
                elif 720/2 < x_center1 <720/2+720/2:
                    x_center2='middle'
                elif 720/2 < x_center1:
                    x_center2='right'
                if y_center1 < 720/2:
                    y_center2='up'
                elif 720/2 < y_center1 <720/2+720/2:
                    y_center2='center'
                elif 720/2 < y_center1:
                    y_center2='down'
            

    # 선택적: 프레임 출력
    cv2.imshow('Frame', det)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
