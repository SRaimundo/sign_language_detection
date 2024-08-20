import cv2
import time
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class_name = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
                      'N', 'O', 'R', 'S', 'T', 'U', 'V', 'W', 'Y'])


class INF692Net(torch.nn.Module):
    def __init__(self):
        super(INF692Net, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)) # 64x64
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)) # 32x32
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2)) # 16x16
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(16*16*128, 64),
            torch.nn.ReLU())
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU())
        self.fc2 = torch.nn.Sequential(
            torch.nn.Dropout(0.25),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU())
        self.fc3= torch.nn.Sequential(
            torch.nn.Dropout(0.25),
            torch.nn.Linear(128, 21))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

data_transform = transforms.Compose([
	transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


model = INF692Net()
model.load_state_dict(torch.load('models/INF692NetHenriqueI10.pth', map_location=torch.device(device)))
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
