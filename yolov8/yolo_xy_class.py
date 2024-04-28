from ultralytics import YOLO
import pytesseract
import cv2
from gtts import gTTS
import pygame

# YOLO 모델 로드
model = YOLO('yolov8n.pt')

# 비디오 파일 로드
cap = cv2.VideoCapture('C:/project/yolov8/video/working2.mp4')


# def text_to_speech(text):
#     if text:
#         tts = gTTS(text=text, lang='en')
#         output_path = 'C:/project/yolov8/sound/xy_class2.mp3'
        # tts.save(output_path)
#         play_sound('C:/project/yolov8/sound/xy_class2.mp3')

# def play_sound(file_path):
#     pygame.mixer.init()  # mixer 초기화
#     pygame.mixer.music.load(file_path)  # 오디오 파일 로드
#     pygame.mixer.music.play()

# 프레임별로 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO를 사용하여 프레임에서 객체 탐지
    results = model.predict(frame, conf=0.7)
    
    confidence_thresholds = {
    'person': 0.7,
    'bicycle': 0.5,
    'car': 0.7,
    # 기타 클래스에 대한 임곗값 설정
}
    # 탐지된 객체들을 순회하면서
    for det in results:
        box = det.boxes
        for i in range(len(box.xyxy)):
            x1, y1, x2, y2 = box.xyxy[i].tolist()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{det.names[i]}:{box.conf[i]:.2f}"
            conf = box.conf[i]
            if conf < confidence_thresholds.get(det.names[i], 0.5):  # 클래스별 임계값 사용
                continue
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            x_center1 = (x1 + x2) / 2
            y_center1 = (y1 + y2) / 2
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
            location = f"{x_center2} {y_center2}"
            print(f"{location}, {label}")
            # text_to_speech(f"{location}, {label}")
            # play_sound(text_to_speech(f"{location}, {label}"))

    # 프레임 출력
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

