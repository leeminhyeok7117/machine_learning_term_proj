import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ✅ 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4  # level 1~4
class_names = ["level 1", "level 2", "level 3", "level 4"]  # 클래스 이름

# ✅ 이미지 경로
image_path = "/home/lmh/machine_learning_term_proj/paper_final/answer/level1.jpg"  # ← 평가할 이미지 경로

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 모델 입력 사이즈에 맞춤
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ 이미지 로드 및 전처리
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)  # (1, C, H, W)

# ✅ 모델 로드
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load("/home/lmh/machine_learning_term_proj/paper_final/best_model2.pth", map_location=device))
model = model.to(device)
model.eval()

# ✅ 예측 및 확률 계산
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
    predicted_class = np.argmax(probs)

# ✅ 출력
print("📊 클래스별 확률:")
for i, (name, prob) in enumerate(zip(class_names, probs)):
    print(f"{name}: {prob*100:.2f}%")
print(f"\n📌 예측 결과: {class_names[predicted_class]}")

# ✅ 시각화 (이미지 + 확률 막대 그래프)
plt.figure(figsize=(10, 4))

# 1. 원본 이미지
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title(f"Predicted: {class_names[predicted_class]}")
plt.axis('off')

# 2. 확률 막대 그래프
plt.subplot(1, 2, 2)
bars = plt.bar(class_names, probs, color='skyblue')
bars[predicted_class].set_color('orange')  # 예측 클래스 강조
plt.ylim(0, 1)
plt.title("Class Probabilities")

# Show everything
plt.tight_layout()
plt.show()
