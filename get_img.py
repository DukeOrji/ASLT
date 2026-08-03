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

if len(X) == 0:
    print("Empty Dataset")


def process():
    
    
    print(len(X))
    input("")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    # y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    print(f"Dataset size: {len(dataset)}")
    server = Server(train_set=None, test_set=loader)

    if os.path.exists("checkpoint.pth"):
        checkpoint = torch.load("checkpoint.pth", map_location=device)

        optim_dict = checkpoint["optimizer_state_dict"]
        model_dict = checkpoint["model_state_dict"]
        server.set_weight(model_dict, optim_dict)

    server.predict_label()
    X.clear()
    #y.clear()



cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    success, frame = cap.read()

    if not success:
        break

    cv2.imshow("FrameCap", frame)
    frame = cv2.resize(frame, (224, 224))
    key = cv2.waitKey(1)

    # if key in classes:
    #     current_label = classes[key]
    #     print(f"Current label: {chr(key).upper()}")

    if key == ord(" "): #space to capture
        for _ in range(5): #increases likelihood of capturing a frame per click
            landmarks = get_landmarks(frame)
            if landmarks is not None:
                X.append(landmarks)
                #y.append(current_label)
                print("Frame Captured.")
                break
            
            else:
                print("No hands detected.")

    elif key == 9:
        process()
        
    elif key == 27: #ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

