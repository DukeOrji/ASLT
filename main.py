#main.py
from ds import asl_dataloader
from server import Server
from config import device


landmarks = asl_dataloader()
server = Server(landmarks)


print(next(server.model.parameters()).device)
for i in range(15):
    print(f"\nround {i+1}")
    print()
    loss, acc = server.train()
    print(f"\nTraining  Loss: {loss}  Acc: {acc}")

    # f_loss, f_acc = server.evaluate()
    # print(f"\nFinal Loss: {f_loss}  Acc: {f_acc}")
    # print("\nExperiment Complete.")