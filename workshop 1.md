```bash

import torch
import os
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import numpy as np
from time import time

def load_mnist(data_dir='~/.pytorch/MNIST_data/', batch_size=64):
    """
    Load MNIST dataset from local directory if exists, otherwise download it.
    
    Parameters:
        data_dir (str): Directory to store/load the dataset
        batch_size (int): Number of samples per batch
        
    Returns:
        tuple: (train_loader, val_loader) containing the data loaders for training and validation
    """
    data_dir = os.path.expanduser(data_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = datasets.MNIST(data_dir, 
                            download=not os.path.exists(os.path.join(data_dir, 'MNIST')),
                            train=True, 
                            transform=transform)
    
    valset = datasets.MNIST(data_dir,
                           download=not os.path.exists(os.path.join(data_dir, 'MNIST')),
                           train=False,
                           transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    
    return trainloader, valloader

def display_sample_images(trainloader, num_images=60):
    """
    Display sample images from the dataset.
    
    Parameters:
        trainloader: DataLoader containing the training data
        num_images (int): Number of images to display
    """
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    
    plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
    figure = plt.figure()
    for index in range(1, num_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    plt.show()
    
    return images, labels

def create_model(input_size=784, hidden_sizes=[128, 64], output_size=10):
    """
    Create a feed-forward neural network model.
    
    Parameters:
        input_size (int): Size of input layer
        hidden_sizes (list): List of hidden layer sizes
        output_size (int): Size of output layer
        
    Returns:
        model: PyTorch neural network model
    """
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[1], output_size),
                         nn.LogSoftmax(dim=1))
    return model

def train_model(model, trainloader, epochs=15, learning_rate=0.003, momentum=0.9):
    """
    Train the neural network model.
    
    Parameters:
        model: PyTorch model to train
        trainloader: DataLoader containing training data
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimization
        momentum (float): Momentum for optimization
        
    Returns:
        model: Trained PyTorch model
        training_time (float): Time taken for training in minutes
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    time0 = time()
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            
            output = model(images.to(device))
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {e} - Training loss: {running_loss/len(trainloader)}")
    
    training_time = (time()-time0)/60
    print(f"\nTraining Time (in minutes) = {training_time}")
    return model, training_time

def evaluate_model(model, valloader):
    """
    Evaluate the model's accuracy on validation data.
    
    Parameters:
        model: Trained PyTorch model
        valloader: DataLoader containing validation data
        
    Returns:
        float: Model accuracy
    """
    correct_count, all_count = 0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for images, labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img.to(device))
            ps = torch.exp(logps)
            probab = list(ps.cpu().numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if true_label == pred_label:
                correct_count += 1
            all_count += 1
    
    accuracy = correct_count/all_count
    print(f"Number Of Images Tested = {all_count}")
    print(f"Model Accuracy = {accuracy}")
    return accuracy

def main():
    """
    Main function to orchestrate the training and evaluation process.
    """
    # Load data
    trainloader, valloader = load_mnist()
    
    # Display sample images
    images, labels = display_sample_images(trainloader)
    
    # Create and train model
    model = create_model()
    model, training_time = train_model(model, trainloader)
    
    # Evaluate model
    accuracy = evaluate_model(model, valloader)
    
    return model, accuracy

if __name__ == "__main__":
    model, accuracy = main()

```
