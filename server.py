import torch.nn as nn
import torch
from config import device
import torch.optim as optim

class ASLNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(
            128,
            256,
            kernel_size=3,
            padding=1
        )
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(
            256,
            512,
            kernel_size=3,
            padding=1
        )
        self.bn5 = nn.BatchNorm2d(512)

        self.fc_layer = nn.Sequential(
            nn.Linear(8192, 712),
            nn.ReLU(),
            nn.Linear(712, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 29)
        )

    def forward(self, x):
        #first convolutional layer - learn textures, shapes, etc.
        x = self.pool(
            torch.relu(
                self.bn1(self.conv1(x))
            )
        )

        x = self.pool(
            torch.relu(
                self.bn2(self.conv2(x))
            )
        )

        x = self.pool(
            torch.relu(
                self.bn3(self.conv3(x))
            )
        )

        x = self.pool(
            torch.relu(
                self.bn4(self.conv4(x))
            )
        )

        x = self.pool(
            torch.relu(
                self.bn5(self.conv5(x))
            )
        )

        x = x.view(x.size(0), -1)  #flatten to 2 dimensional vector 
        x = self.fc_layer(x)
        return x

class Server():
    def __init__(self, dataloader, testloader):
        self.model = ASLNet()
        self.dataloader = dataloader
        self.testloader = testloader
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)
    
    def train(self):
        correct = 0
        total = 0
        losses = []

        for batch_idx, (images, labels) in enumerate(self.dataloader):
            if batch_idx == 32:
                break

            images = images.to(device)
            labels = labels.to(device)

            pred = self.model(images)
            pred_labels = pred.argmax(dim=1)

            loss = self.loss_fn(pred, labels)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step
            losses.append(loss.item())

            print(f"Img{batch_idx}  Loss: {loss}")
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

        acc = round(correct/total, 2)
        avg_loss = round(sum(losses)/len(losses), 2)

        return avg_loss, acc
    
    def evaluate(self):
        total = 0
        correct = 0

        losses = []
        
        with torch.no_grad():
            self.model.eval()
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images = images.to(device)
                labels = labels.to(device)
                pred = self.global_model(norm(images))
                pred_labels = pred.argmax(dim=1)

                loss = self.loss_fn(pred, labels)
                losses.append(loss.item())
                correct += (pred_labels == labels).sum().item()
                total += labels.size(0)

            acc = round(correct/total, 2)
            avg_loss = round(mean(losses), 3)
        
        return avg_loss, acc