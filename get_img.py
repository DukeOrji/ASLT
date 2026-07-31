import cv2
import os
import numpy
from mpipeline import get_landmarks
from server import Server
import torch
from config import device
from torch.utils.data import TensorDataset, DataLoader
from class_names import classes



X = []
y = []
if len(X) == 0:
    print("Empty Dataset")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    success, frame = cap.read()

    if not success:
        break

    cv2.imshow("FrameCap", frame)
    frame = cv2.resize(frame, (224, 224))
    key = cv2.waitKey(1)

    if key in classes:
        current_label = classes[key]
        print(f"Current label: {chr(key).upper()}")

    elif key == ord(" "): #space to capture

        landmarks = get_landmarks(frame)
        if landmarks is not None:
            X.append(landmarks)
            y.append(current_label)
            print("Frame Captured.")
            
        else:
            print("No hands detected.")
        
    

    elif key == 27: #ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

print(len(y))
print(len(X))
input("")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

print(f"Dataset size: {len(dataset)}")
server = Server(train_set=None, test_set=loader)

if os.path.exists("checkpoint.pth"):
    checkpoint = torch.load("checkpoint.pth", map_location=device)

    optim_dict = checkpoint["optimizer_state_dict"]
    model_dict = checkpoint["model_state_dict"]
    server.set_weight(model_dict, optim_dict)

server.predict_label()