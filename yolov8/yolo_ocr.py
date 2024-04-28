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
    results = model.predict(frame, conf=0.9)
    
    # 탐지된 객체들을 순회하면서
    for det in results:
        # det.boxes.numpy()를 사용하여 모든 경계 상자 정보를 numpy 배열로 변환
        if det.boxes:
            bboxes = det.boxes.cpu()  # bboxes는 모든 경계 상자를 포함하는 numpy 배열
            for box in bboxes:
                x1, y1, x2, y2, _, _ = box  # 각 box는 [x1, y1, x2, y2, conf, cls] 형태의 배열
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # 탐지된 객체 영역을 이미지에서 추출
                roi = frame[y1:y2, x1:x2]

                # pytesseract를 사용하여 ROI에서 텍스트 추출
                text = pytesseract.image_to_string(roi)
                print(text)

                # 선택적: 탐지된 텍스트를 출력
                print(text)

    # 선택적: 프레임 출력
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
boxes = r.boxes  # Boxes object for bbox outputs
masks = r.masks  # Masks object for segment masks outputs
probs = r.probs  # Class probabilities for classification outputs
'''

