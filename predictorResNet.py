import cv2
import time
import numpy as np
import torch
from PIL import Image
import torchvision
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class_name = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
                      'N', 'O', 'R', 'S', 'T', 'U', 'V', 'W', 'Y'])


class INF692Net(torch.nn.Module):
    def __init__(self):
        super(INF692Net, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_features, 21)
        

    def forward(self, x):
        out = self.resnet(x)
        return out


data_transform = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()

model = INF692Net()
model.load_state_dict(torch.load('models/INF692NetResNET18.pth', map_location=torch.device(device)))
model.eval()


def predict_label(image):
    image = cv2.resize(image, (224, 224)) 
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
    img_transformed = data_transform(image)
    img_tensor = img_transformed.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)
    label = predicted.item()
    
    return label


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  
    if not ret:
        break

    label = predict_label(frame)
    
    cv2.putText(frame, f'Predicted: {class_name[label]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Camera', frame)
    
    time.sleep(0.05)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
