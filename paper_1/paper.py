import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ✅ 하이퍼파라미터
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 5
IMG_SIZE = 224

# ✅ 데이터 경로
train_dir = 'dataset/train'
val_dir = 'dataset/val'

# ✅ 데이터 전처리 및 Augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean/std
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ 데이터셋
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ✅ 모델 구성 (ResNet18 사용)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ✅ 손실함수 및 최적화기
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ 학습 함수
def train():
    model.train()
    total_loss = 0
    correct = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()

    acc = correct / len(train_dataset)
    print(f"[Train] Loss: {total_loss:.4f} | Acc: {acc:.4f}")

# ✅ 검증 함수
def validate():
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

    acc = correct / len(val_dataset)
    print(f"[Val] Loss: {total_loss:.4f} | Acc: {acc:.4f}")

# ✅ 전체 학습 루프
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train()
    validate()

# ✅ 모델 저장
torch.save(model.state_dict(), 'folding_classifier.pth')
print("✅ 모델 저장 완료: folding_classifier.pth")
