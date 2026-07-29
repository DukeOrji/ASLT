#server.py
import torch.nn as nn
import torch
from models import MobileNetV3, ASLNet, LandmarkNet
from config import device, num_batch
import torch.optim as optim


jms = []

class Server():
    def __init__(self, landmarks):
        self.model = LandmarkNet()
        self.model = self.model.to(device)
        self.landmarks = landmarks
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)
        
    
    def train(self):
        self.model.train()
        correct = 0
        total = 0
        losses = []

        for batch_idx, (lm, labels) in enumerate(self.landmarks):
            # if batch_idx == num_batch:
            #     break

            landmarks = lm.to(device)
            labels = labels.to(device)

            pred = self.model(landmarks)
            pred_labels = pred.argmax(dim=1)

            loss = self.loss_fn(pred, labels)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses.append(loss.item())

            if batch_idx % 5 == 0:
                print(f"Img{batch_idx}  Loss: {loss}")

            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

        acc = round(correct/total, 3)
        avg_loss = round(sum(losses)/len(losses), 3)
        jms.append(self.model.state_dict())

        return avg_loss, acc
    
    def evaluate(self):

        self.model.load_state_dict(jms[0])
        del jms[0]

        total = 0
        correct = 0

        losses = []
        
        with torch.no_grad():

            for batch_idx, (images, labels) in enumerate(self.testloader):
                
                images = images.to(device)
                labels = labels.to(device)
                pred = self.model(images)
                pred_labels = pred.argmax(dim=1)

                loss = self.loss_fn(pred, labels)
                losses.append(loss.item())

                # if batch_idx % 5 == 0:
                #     print(f"Img{batch_idx}  Loss: {loss}")
                correct += (pred_labels == labels).sum().item()
                total += labels.size(0)

            acc = round(correct/total, 2)
            avg_loss = round(sum(losses)/len(losses), 3)
        
        return avg_loss, acc

