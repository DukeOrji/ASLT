import cv2
import os
import numpy
from mpipeline import get_landmarks
from server import Server
import torch
from config import device
from torch.utils.data import TensorDataset, DataLoader

classes = {
    ord("a"): 0,
    ord("b"): 1,
    ord("c"): 2,
    ord("d"): 3,
    ord("e"): 4,
    ord("f"): 5,
    ord("g"): 6,
    ord("h"): 7,
    ord("i"): 8,
    ord("j"): 9,
    ord("k"): 10,
    ord("l"): 11,
    ord("m"): 12,
    ord("n"): 13,
    ord("o"): 14,
    ord("p"): 15,
    ord("q"): 16,
    ord("r"): 17,
    ord("s"): 18,
    ord("t"): 19,
    ord("u"): 20,
    ord("v"): 21,
    ord("w"): 22,
    ord("x"): 23,
    ord("y"): 24,
    ord("z"): 25,
}


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

loss, acc = server.evaluate()
print(f"Evaluation Loss: {loss}  |  Acc: {acc}")