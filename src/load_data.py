import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_data(path):
    # Define the transformations (resize, normalize, etc.)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  
        transforms.ToTensor(),        
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(root=path, transform=transform)

    # Calculate the sizes for the train and test datasets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset into train and test sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    print(f'Size of the training dataloader: {len(train_loader)}')
    print(f'Size of the test dataloader: {len(test_loader)}')

    # Return both loaders and the original dataset classes
    return train_loader, test_loader, dataset.classes
