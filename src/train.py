import torch
import torch.nn as nn
import torch.optim as optim
import os

def train(model,data_loader,test_loader,device,epochs,optimizer,criterion):
    '''
    Train the model 
    Inputs: 
    learning rate,
    epochs: number of epochs
    returns the trained mddel 
    '''
    directory = 'Saved Model'

    # Creating the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    model.to(device)
    
    print('--------------------Training Started---------------------')
# Training loop
    best_loss = float('inf')
    train_loss_history = list()
    train_accuracy_history = list()

    test_loss_history = list()
    test_accuracy_history = list()
    
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        # Print loss and accuracy for this epoch
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {running_loss / len(data_loader):.4f}, Train Accuracy: {accuracy:.2f}%')
        train_loss_history.append(running_loss / len(data_loader))
        train_accuracy_history.append(accuracy)
        val_loss = 0.0
        val_correct = 0
        model.eval()
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            _,preds = torch.max(outputs,1)
            val_correct += (preds==labels).float().mean().item()
            val_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Test Loss: {val_loss/len(test_loader):.5f}, Test Accuracy: {val_correct/len(test_loader)*100:.2f}% ')
        test_loss_history.append(val_loss/len(test_loader))
        test_accuracy_history.append(val_correct/len(test_loader)*100)
        if loss < best_loss:
            best_loss = loss
        best_state_dict = model.state_dict()
    print('--------------------Training Ended ----------------------')
    torch.save(best_state_dict, os.path.join(directory, 'best_model.pth'))    
    return [train_accuracy_history,test_accuracy_history,train_loss_history,test_loss_history]
