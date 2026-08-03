import cv2
import os
import numpy
from mpipeline import get_landmarks
from server import Server
import torch
from config import device
from torch.utils.data import TensorDataset, DataLoader
from class_names import classes







def get_server():
    server = Server(
        train_set=None,
        test_set=None
    )

    if os.path.exists("checkpoint.pth"):
        checkpoint = torch.load("checkpoint.pth", map_location=device)

        optim_dict = checkpoint["optimizer_state_dict"]
        model_dict = checkpoint["model_state_dict"]
        server.set_weight(model_dict, optim_dict)

    return server


def process(server, landmarks):
    

    data = torch.tensor(landmarks, dtype=torch.float32, device=device).unsqueeze(0)

    p_label, confidence = server.predict_label(data)
    if p_label == "space":
        p_label = " "
    
    return p_label, confidence


current_word = []
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
server = get_server()
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
            success, frame = cap.read()
            if not success:
                break

            landmarks = get_landmarks(frame)
            if landmarks is not None:
                
                #print("Frame Captured.")
                p_label, confidence = process(server, landmarks)
                print(f"-- {p_label}  | {confidence}")
                current_word.append(p_label)

                break
            
            else:
                print("No hands detected.")

    elif key == 8: #backspace key
        if current_word:
            del current_word[-1]
        print('\n', "".join(current_word))
        
    elif key == 27: #ESC to quit
        print(f"Last word formed: {''.join(current_word)}")
        break

cap.release()
cv2.destroyAllWindows()

