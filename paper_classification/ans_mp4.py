import cv2
import torch
import numpy as np
from torchvision import models, transforms

# ✅ 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
class_names = ["level 1", "level 2", "level 3", "level 4"]

# ✅ 모델 로드
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load("/Users/USER/Desktop/paper_final/best_model3.pth", map_location=device))
model = model.to(device)
model.eval()

# ✅ 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ 웹캠 열기
cap = cv2.VideoCapture(1)  # 0번 카메라

if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ 프레임 복사 (화면 출력용)
    display_frame = frame.copy()

    # ✅ 이미지 전처리
    input_tensor = transform(frame).unsqueeze(0).to(device)

    # ✅ 예측
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
        predicted_class = np.argmax(probs)

    # ✅ 결과 출력
    label = f"{class_names[predicted_class]} ({probs[predicted_class]*100:.1f}%)"
    cv2.putText(display_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # ✅ 확률 막대 표시 (좌측 상단부터 쭉)
    for i, (cls, p) in enumerate(zip(class_names, probs)):
        text = f"{cls}: {p*100:.1f}%"
        y = 60 + i * 30
        color = (0, 128, 255) if i == predicted_class else (200, 200, 200)
        cv2.putText(display_frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ✅ 화면 표시
    cv2.imshow("Paper Folding Stage Classification", display_frame)

    # 종료 조건: q 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ 정리
cap.release()
cv2.destroyAllWindows()
