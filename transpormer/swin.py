import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
from torch.utils.data import DataLoader

# 데이터셋 준비
transform = transforms.Compose([
    transforms.Resize(224),  # Swin Transformer 모델은 224x224 크기의 이미지를 입력으로 사용합니다.
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Swin Transformer 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=10)
model.to(device)

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 함수
def train_model(model, train_loader):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)  # Swin Transformer 모델의 출력은 logits 속성에 없습니다.
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Training Loss: {total_loss / len(train_loader)}')

# 테스트 함수
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Swin Transformer 모델의 출력은 logits 속성에 없습니다.
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')

# 학습 및 테스트
num_epochs = 1
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    train_model(model, train_loader)
    test_model(model, test_loader)
