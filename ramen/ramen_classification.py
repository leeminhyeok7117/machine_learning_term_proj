import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ============================================
# 설정
# ============================================
data_dir = './dataset'  # 'train'과 'test'만 필요
num_classes = 4
batch_size = 32
num_epochs = 50
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = './checkpoints'
os.makedirs(save_dir, exist_ok=True)

# TensorBoard 로그 디렉토리
writer = SummaryWriter(log_dir='./runs/exp1')

# ============================================
# 데이터 전처리
# ============================================
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# ============================================
# 데이터셋 및 DataLoader
# ============================================
image_datasets = {
    phase: datasets.ImageFolder(os.path.join(data_dir, phase), transform[phase])
    for phase in ['train', 'test']
}
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2),
    'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=2)
}
class_names = image_datasets['train'].classes
print(f"🔖 클래스: {class_names}")

# ============================================
# EfficientNet 모델 준비
# ============================================
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model = model.to(device)

# ============================================
# 클래스 가중치 및 손실함수
# ============================================
targets = [label for _, label in image_datasets['train'].imgs]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(targets), y=targets)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"클래스 가중치: {class_weights_tensor}")

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

# ============================================
# 학습 루프 + 매 epoch test
# ============================================
for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")

    # ========== Train ==========
    model.train()
    running_loss = 0.0
    running_corrects = 0
    loop = tqdm(dataloaders['train'], desc="Train")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(image_datasets['train'])
    epoch_acc = running_corrects.double() / len(image_datasets['train'])
    print(f"Train | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
    writer.add_scalar("train/Loss", epoch_loss, epoch)
    writer.add_scalar("train/Accuracy", epoch_acc, epoch)

    # ========== Test (매 epoch마다) ==========
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_correct += torch.sum(preds == labels).item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total
    print(f"🧪 Test Accuracy: {test_acc:.4f} ({test_correct}/{test_total})")
    writer.add_scalar("test/Accuracy", test_acc, epoch)

    # 모델 저장
    torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1:02d}.pth"))
    print(f"모델 저장됨: epoch_{epoch+1:02d}.pth")

    scheduler.step()

# 최종 모델 저장
torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_model.pth"))
print("최종 모델 저장됨: last_epoch_model.pth")

writer.close()






# ============================================
# ✅ 예측 함수
# ============================================
def predict_image(image_path, model, class_names):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)[0]
        _, pred = torch.max(outputs, 1)

    print(f"예측 클래스: {class_names[pred.item()]}")
    print(f"확률 분포:")
    for i, prob in enumerate(probabilities):
        print(f"  {class_names[i]}: {prob.item() * 100:.2f}%")

    return class_names[pred.item()]




model.load_state_dict(torch.load(os.path.join(save_dir, "last_epoch_model.pth")))
print("예측 결과:", predict_image("kko3.png", model, class_names))
