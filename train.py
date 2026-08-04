#main.py
from ds import asl_dataloader
import torch
from server import Server
from config import device


train_set, test_set = asl_dataloader()
server = Server(train_set, test_set)


print(next(server.model.parameters()).device)
for i in range(25):
    print(f"\nround {i+1}")
    loss, acc = server.train()
    print(f"\nTraining  Loss: {loss}  Acc: {acc}")

    f_loss, f_acc = server.evaluate()
    print(f"Final Loss: {f_loss}  Acc: {f_acc}")
print("\nExperiment Complete.")



model_dict, optim_dict = server.broadcast_weight()
torch.save({
    "model_state_dict": model_dict,
    "optimizer_state_dict": optim_dict
}, "checkpoint.pth")