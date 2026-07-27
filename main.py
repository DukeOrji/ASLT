from ds import load_asl_dataset
from server import Server


dataloader, testloader = load_asl_dataset()
server = Server(dataloader, testloader)


for range in range(15):
    print()
    server.train()

    # f_loss, f_acc = server.evaluate()
    # print(f"F Loss: {f_loss}   F Acc: {f_acc}")