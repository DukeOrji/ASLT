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
        range(len(dataset))
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

    

def asl_dataloader():
    #load_asl_dataset()
    X, y = torch.load("landmarks.pt")

    print(X.shape)
    print(y.shape)

    dataset = TensorDataset(X, y)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(
        dataset,
        [train_size, test_size]
    )

    loader = DataLoader(
        train_set,
        batch_size=64,
        shuffle=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=64,
        shuffle=False
    )

    return loader, test_loader