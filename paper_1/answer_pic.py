import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image

# ======== 설정 ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5  # 클래스 개수 (dataset에 맞게 조정)
model_path = "/home/lmh/machine_learning_term_proj/paper/paper_fold_classifier.pth"  # 학습된 모델 경로

# ======== 모델 정의 ========
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model = model.to(device)

# ======== 모델 가중치 로드 ========
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ======== 클래스 이름 (ImageFolder 기준) ========
class_names = ['level1', 'level2', 'level3', 'level4', 'level5']  # dataset.classes 로 변경 가능

# ======== 예측 함수 ========
def predict_image(image_path, model, class_names, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)

    print(f"예측 클래스: {class_names[predicted.item()]}")
    print("확률 분포:")
    for i, prob in enumerate(probabilities):
        print(f"  {class_names[i]}: {prob.item()*100:.2f}%")

    return class_names[predicted.item()]

# ======== 예측 실행 ========
image_path = "level5_2.jpg"  # 예측할 이미지 경로로 변경
result = predict_image(image_path, model, class_names, device)
print("최종 예측 결과:", result)
