import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import numpy as np

# ✅ 클래스 이름 (label 순서)
class_names = ["level1", "level2", "level3", "level4", "level5"]

# ✅ 모델 구조 정의 (저장할 때 사용한 구조와 동일해야 함)
num_classes = 5
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# ✅ state_dict 로드
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# ✅ 전처리 정의
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),   # Roboflow 권장 사이즈
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 기반
                         std=[0.229, 0.224, 0.225])
])

# ✅ 웹캠 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 입력 없음")
        break

    input_img = transform(frame).unsqueeze(0)  # 배치 차원 추가

    with torch.no_grad():
        output = model(input_img)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]

    # 화면에 예측 결과 표시
    cv2.putText(frame, f"Predicted: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Real-time Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
