import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from google.colab import files
import os
import zipfile
from tqdm import tqdm

uploaded = files.upload()  

# 압축 해제
zip_path = "/content/dataset_second.zip"
extract_path = "/content/dataset_second"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# CBAM 모듈 정의
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        x = x * self.sigmoid_channel(avg_out + max_out)
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg, max_], dim=1)
        spatial_attn = self.sigmoid_spatial(self.conv_spatial(concat))
        x = x * spatial_attn

        return x

# EfficientNet + CBAM 정의
class EfficientNetCBAM(nn.Module):
    def __init__(self, num_classes=4, cbam_block_idx=2):
        super(EfficientNetCBAM, self).__init__()
        self.backbone = efficientnet_b0(pretrained=True)
        self.features = nn.Sequential()
        for i, block in enumerate(self.backbone.features):
            self.features.add_module(f'block{i}', block)
            if i == cbam_block_idx:
                out_channels = block[0].out_channels
                self.features.add_module(f'cbam{i}', CBAM(out_channels))

        in_features = self.backbone.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x

# 학습 파라미터 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4
batch_size = 32
epochs = 15
learning_rate = 1e-4
best_acc = 0.0

# 데이터셋 불러오기
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dir = os.path.join(extract_path, "train")
valid_dir = os.path.join(extract_path, "valid")
test_dir = os.path.join(extract_path, "test")

train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 모델 정의
model = EfficientNetCBAM(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training", leave=False)
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    model.eval()
    correct, total = 0, 0
    loop = tqdm(valid_loader, desc=f"Epoch [{epoch+1}/{epochs}] Validation", leave=False)
    with torch.no_grad():
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(acc=correct/total)

    acc = correct / total
    print(f"Validation Accuracy: {acc*100:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model_cbam.pth")
        print("Best model saved.")

print("최종 test셋 검증 시작")
model.load_state_dict(torch.load("best_model_cbam.pth"))
model.eval()
correct, total = 0, 0
loop = tqdm(test_loader, desc="Testing")
with torch.no_grad():
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(acc=correct/total)

final_acc = correct / total
print(f"최종 Test Accuracy (best model): {final_acc*100:.2f}%")