import torch

def eval(model,test_loader,device):
 
    model.eval()
    total = 0
    correct = 0  
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the test set:  { (100 * correct / total):.2f}%')
