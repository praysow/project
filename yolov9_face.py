import os
import torch
from ultralytics import YOLO
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
# 환경 변수 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 경로 설정
data_path = 'c:/datasets/class/'
train_img_path = os.path.join(data_path, 'images', 'train')
val_img_path = os.path.join(data_path, 'images', 'val')
train_label_path = os.path.join(data_path, 'labels', 'train')
val_label_path = os.path.join(data_path, 'labels', 'val')

# YOLOv8 모델 초기화
model = YOLO('yolov9c.pt')  # YOLOv8 모델 사용

# 데이터셋 정의를 위한 YAML 파일 경로
data_yaml_path = 'c:/datasets/yolo_face.yaml'

# 하이퍼파라미터 설정 (옵션)
hyp = {
    'lr0': 0.01,  # 초기 학습률
    'epochs': 500,  # 총 에포크 수
    'batch_size': 16,  # 배치 크기
    'img_size': 640,  # 이미지 크기
    'patience': 30  # early stop을 위한 인내심 값

}

# 모델 훈련
best_mAP = 0.0
patience_counter = 0
for epoch in range(hyp['epochs']):
    model.train(data=data_yaml_path, epochs=500, batch=hyp['batch_size'], imgsz=hyp['img_size'], lr0=hyp['lr0'])

    # 훈련된 모델 성능 평가
    results = model.val(data=data_yaml_path, batch_size=hyp['batch_size'], imgsz=hyp['img_size'])

    # mAP 값을 결과에서 추출
    mAP = results.metrics.mAP_0_5_0_95  # 변경된 mAP 값 접근 방식 확인 필요

    print(f"Epoch [{epoch + 1}/{hyp['epochs']}], mAP: {mAP}")

    # 성능이 개선되지 않을 때마다 patience_counter를 증가시킴
    if mAP > best_mAP:
        best_mAP = mAP
        patience_counter = 0
        # 훈련된 모델 저장
        model.save('c:/yolo/face/yolo_face.pt')  # 절대 경로로 저장
    else:
        patience_counter += 1

    # patience_counter가 인내심 값보다 크거나 같으면 조기 종료
    if patience_counter >= hyp['patience']:
        print(f"Early stopping at epoch {epoch + 1}")
        break
