
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


api_key = "KGAT_403dfa0d85a7eeead4d5d50ee9ce505a"



def load_asl_dataset():
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    dataset = datasets.ImageFolder(
        root="./asl_alphabet_train",
        transform=preprocess
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(
        dataset,
        [train_size, test_size]
    )

    dataloader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=64
    )

    testloader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=64
    )

    return dataloader, testloader