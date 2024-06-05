import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import cv2
import random

# COCO 클래스
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Mask R-CNN 모델 불러오기
model = maskrcnn_resnet50_fpn_v2(pretrained=True)
model.eval()


def predict(image_path):
    img = Image.open(image_path)
    img_tensor = F.to_tensor(img)
    with torch.no_grad():
        predictions = model([img_tensor])
    return predictions


def visualize(image_path, predictions):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for score, label, mask, box in zip(predictions[0]['scores'], predictions[0]['labels'], predictions[0]['masks'],
                                       predictions[0]['boxes']):
        if score > 0.5:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            mask = mask[0, :, :].numpy()
            mask = (mask > 0.5).astype(np.uint8)  # 이진화 처리

            # 마스크를 반투명하게 적용
            mask_img = img.copy()  # 원본 이미지의 복사본 생성
            mask_img[mask == 1] = (mask_img[mask == 1] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

            img[mask == 1] = mask_img[mask == 1]  # 반투명 마스크 영역 적용

            # 바운딩 박스 그리기
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            cv2.rectangle(img, start_point, end_point, color, 3)

            # 클래스 이름 표시
            cv2.putText(img, COCO_INSTANCE_CATEGORY_NAMES[label], (int(box[0]), int(box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 이미지 출력
    img = Image.fromarray(img)
    img.show()


# 이미지 경로 설정 및 예측 수행
image_path = '/home/aia/yolo_sw/img/123.jpg'
# 예측 수행
predictions = predict(image_path)

# 시각화
visualize(image_path, predictions)
