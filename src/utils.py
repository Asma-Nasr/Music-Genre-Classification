import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

def plot_accuracy_loss(history,mode):
    if mode == 'loss':
        plt.plot(history[2],label='training loss')
        plt.plot(history[3],label='testing loss')
        plt.legend()
        plt.show()
    elif mode == 'accuracy':
        plt.plot(history[0],label='training accuracy')
        plt.plot(history[1],label='testing accuracy')
        plt.legend()
        plt.show()
    else:
        raise NameError(f"Unknown mode: {mode}, it should be 'loss' or 'accuracy'")
# Evaluate the model
def evaluate_model(model, test_loader,device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = nn.Softmax(dim=1)(outputs)  # Get probabilities
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probabilities.cpu().numpy())  # Store probabilities
    
    return np.array(all_labels), np.array(all_preds)
    
def plot_confusion_matrix(y_true, y_pred, classes):
    '''
    Function to plot the confusion matrix
    Inputs: 
        y_true : true labels
        y_pred : predicted labels
        classes: classes names
    Output:
        Figure shows the confusion matrix of a model ^-^
    '''
    y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

def plot_classification_report(y_true, y_pred):
    y_pred_labels = np.argmax(y_pred, axis=1) 
    print(classification_report(y_true, y_pred_labels))
    
