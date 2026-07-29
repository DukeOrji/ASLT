#models.py
import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)


class MobileNetV3(nn.Module):

    def __init__(
        self,
        num_classes: int = 29,
        pretrained_weights: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        weights = (
            MobileNet_V3_Large_Weights.DEFAULT
            if pretrained_weights
            else None
        )

        backbone = mobilenet_v3_large(weights=weights)

        # Get the input features of the classifier
        in_features = backbone.classifier[0].in_features

        # Freeze/unfreeze the backbone
        for p in backbone.parameters():
            p.requires_grad = not freeze_backbone

        # Remove the original classifier
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # New classification head
        self.head = nn.Linear(in_features, num_classes)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)



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
            nn.Linear(25088, 1000),
            nn.ReLU(),
            nn.Linear(1000, 712),
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


class LandmarkNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(63,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,29)
        )

    def forward(self,x):
        return self.model(x)