#ds.py
from PIL import Image
import numpy as np
import torch
import cv2
from config import num_batch
from torchvision import datasets, transforms
from mpipeline import get_landmarks
from torch.utils.data import Subset, random_split, DataLoader, TensorDataset


api_key = "KGAT_403dfa0d85a7eeead4d5d50ee9ce505a"



def load_asl_dataset():
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
    ])

        

    dataset = datasets.ImageFolder(
        root="./asl_alphabet_train",
        transform=preprocess
    )

    dataset = Subset(
        dataset,
        range(300)
    )

    X = []
    y = []
    for batch_idx, (img, label) in enumerate(dataset):
        # if batch_idx == num_batch:
        #     break
        img = np.array(img)
        landmark = get_landmarks(img)
        if landmark is None:
            continue

        print(f"{batch_idx} ")
        
        X.append(landmark)
        y.append(label)

    torch.save((
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    ), "landmarks.pt")

    return X, y

def asl_dataloader():
    X, y = load_asl_dataset()
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    return loader