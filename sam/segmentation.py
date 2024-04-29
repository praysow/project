from torchvision.models.segmentation import fcn_resnet101
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 이미지 로드
image_path = '/home/aia/yolo/data/images/bus.jpg'
image = Image.open(image_path).convert("RGB")

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)  # 배치 차원 추가

# 모델 로드
model = fcn_resnet101(pretrained=True)

# 모델을 평가 모드로 설정
model.eval()

# 세그멘테이션 수행
with torch.no_grad():
    output = model(input_batch)["out"][0]
output_predictions = output.argmax(0)

# 시각화를 위해 PIL 이미지로 변환
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1, 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(image.size)
r.putpalette(colors)

# 원본 이미지와 세그멘테이션 결과를 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(r)
plt.title("Segmentation Result")
plt.axis("off")

plt.show()
