# import cv2
# from ultralytics import YOLO

# # YOLO 모델 초기화
# model = YOLO("yolov5s.pt")

# def save_frames(video_path, output_path):
#     # 동영상에서 객체 감지
    
#     # 프레임 수
#     num_frames = len(video_path)
#     print("감지된 프레임 수:", num_frames)

#     # 프레임을 이미지로 저장
#     for i, result in enumerate(results):
#         # 프레임 이미지 추출
#         # frame_image = result.orig_img
#         frame_image = result.boxes
        

#         # 이미지 파일로 저장
#         image_path = f"{output_path}/frame_{i}.jpg"
#         cv2.imwrite(image_path, frame_image)
#         print(f"프레임 {i} 저장 완료")

# if __name__ == "__main__":
#     video_path = "c:/Users/bitcamp/Desktop/attention_tensor_dance.mp4"
#     output_path = "c:/Users/bitcamp/Desktop/논문"
#     save_frames(video_path, output_path)

import cv2
import os

# 비디오 파일 경로
video_path = "c:/Users/bitcamp/Desktop/attention_process.mp4"
# 프레임 저장 경로
output_dir = "./dwonf"

# 디렉토리가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

frame_count = 0
while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 프레임 저장
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
    cv2.imwrite(frame_filename, frame)
    
    frame_count += 1

# 비디오 파일 닫기
cap.release()

print(f'Total {frame_count} frames saved to {output_dir}')
