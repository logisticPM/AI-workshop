# MNIST Digit Classification with PyTorch

Welcome to the **MNIST Digit Classification with PyTorch** project! This repository provides a comprehensive guide and implementation for building, training, and evaluating a neural network model to classify handwritten digits from the MNIST dataset using PyTorch. Whether you're a beginner in deep learning or looking to reinforce your understanding of neural networks, this project offers valuable insights and practical experience.

![MNIST Digits](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Understanding the Code](#understanding-the-code)
   - [Imports](#imports)
   - [Loading the MNIST Dataset](#loading-the-mnist-dataset)
   - [Visualizing the Data](#visualizing-the-data)
   - [Building the Neural Network Model](#building-the-neural-network-model)
   - [Training the Model](#training-the-model)
   - [Evaluating the Model](#evaluating-the-model)
   - [Main Function](#main-function)
6. [Running the Project](#running-the-project)
7. [Results](#results)
8. [Conclusion](#conclusion)
9. [References](#references)
10. [License](#license)

---

## Introduction

The **MNIST (Modified National Institute of Standards and Technology)** dataset is a cornerstone in the field of machine learning and computer vision. It comprises 70,000 grayscale images of handwritten digits (0-9), each sized at 28x28 pixels. This project leverages PyTorch, a powerful and flexible deep learning framework, to build a neural network that can accurately classify these digits.

By following this project, you'll gain hands-on experience with:

- **Data Loading and Preprocessing**: Handling and transforming image data for model training.
- **Neural Network Construction**: Designing a feed-forward neural network with multiple layers.
- **Model Training**: Optimizing the network using gradient descent techniques.
- **Model Evaluation**: Assessing the performance of the trained model on unseen data.

## Prerequisites

Before diving into the project, ensure you have the following installed on your system:

- **Python 3.7 or higher**: [Download Python](https://www.python.org/downloads/)
- **PyTorch**: Installation instructions can be found [here](https://pytorch.org/get-started/locally/).
- **Torchvision**: Comes bundled with PyTorch, but ensure it's installed.
- **Matplotlib**: For data visualization.
- **NumPy**: For numerical operations.
- **Git**: For version control and repository management.

You can install the required Python libraries using `pip`:

```bash
pip install torch torchvision matplotlib numpy
```
## Import
```bash
import torch
import os
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import numpy as np
from time import time**
```
Explanation:

- **torch**: Core PyTorch library for tensor computations and neural network operations.
- **os**: Interacts with the operating system, handling file paths.
- **torchvision.datasets & transforms**: Utilities for handling image datasets and applying transformations.
- **matplotlib.pyplot**: For plotting and visualizing data.
- **torch.nn**: Provides neural network layers and loss functions.
- **torch.optim**: Offers optimization algorithms for training.
- **numpy**: Facilitates numerical operations on arrays.
- **time**: Measures the duration of training.

## Loading the MNIST Dataset
```bash
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
    
    trainset = datasets.MNIST(
        data_dir, 
        download=not os.path.exists(os.path.join(data_dir, 'MNIST')),
        train=True, 
        transform=transform
    )
    
    valset = datasets.MNIST(
        data_dir,
        download=not os.path.exists(os.path.join(data_dir, 'MNIST')),
        train=False,
        transform=transform
    )
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    
    return trainloader, valloader
```
**Explanation**:

Function Purpose: Loads the MNIST dataset, downloading it if it's not already present locally. It returns DataLoaders for both training and validation datasets.

Parameters:

- **data_dir**: Specifies where to store or load the MNIST data.
- **batch_size**: Determines the number of samples per batch during training and validation.

Data Transformation:

- **ToTensor**: Converts PIL images or NumPy arrays into PyTorch tensors.
- **Normalize**: Standardizes the dataset by adjusting the mean and standard deviation. Here, both are set to 0.5 for grayscale images.
- 
**Dataset Downloading**:

Checks if the MNIST data exists locally. If not, it downloads the dataset.
train=True loads the training set, while train=False loads the validation set.

**DataLoaders**:

DataLoader wraps the dataset and provides batching, shuffling, and parallel data loading.
shuffle=True ensures the data is randomized, enhancing the model's ability to generalize.

Underlying Principles:

- **Data Normalization**: Scaling input data helps in faster convergence and stabilizes the training process by ensuring that all input features contribute equally.
- **Batching**: Processing data in batches improves computational efficiency and allows the model to update its weights more frequently.

## Visualizing the Data
```bash
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
```
**Explanation:**
Function Purpose: Visualizes a set number of sample images from the training dataset to provide an intuitive understanding of the data.

**Parameters**:

-**trainloader**: The DataLoader containing training images.
-**num_images**: The number of images to display in the grid.

**Process**:

Retrieves a batch of images and their corresponding labels.
Uses matplotlib to plot the images in a grid format.
cmap='gray_r' sets the color map to a reversed grayscale, enhancing contrast.

**Visualization Layout**:

The grid is structured as 6 rows by 10 columns to accommodate 60 images.
Each subplot removes axis ticks for a cleaner look.

**Underlying Principles**:

**Data Visualization**: Essential for understanding the dataset's characteristics, spotting anomalies, and verifying the data loading process.
**Grid Display**: Efficiently showcases multiple images simultaneously, allowing for quick inspection of data variety and quality.

## Building the Neural Network Model
```bash
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
    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], output_size),
        nn.LogSoftmax(dim=1)
    )
    return model
```
**Explanation**:

**Function Purpose**: Constructs a feed-forward neural network (also known as a multilayer perceptron) using PyTorch's nn.Sequential container.

**Parameters**:

-**input_size**: Number of input features (28x28 pixels = 784).
-**hidden_sizes**: List specifying the number of neurons in each hidden layer.
-**output_size**: Number of output classes (10 digits).

**Model Architecture**:

-**First Layer**: Connects the input layer to the first hidden layer.
-**Activation Function (ReLU)**: Introduces non-linearity, allowing the network to learn complex patterns.
-**Second Layer**: Connects the first hidden layer to the second hidden layer.
-**Activation Function (ReLU)**: Further non-linearity.
-**Output Layer**: Connects the second hidden layer to the output layer.
-**Output Activation (LogSoftmax)**: Converts raw scores into log-probabilities, which are useful for classification tasks.

**Underlying Principles**:

-**Feed-Forward Neural Networks**: A type of neural network where connections between nodes do not form cycles. They are foundational for many deep learning models.
-**Activation Functions**: Introduce non-linearity into the model, enabling it to learn and model complex data patterns.
-**ReLU (Rectified Linear Unit)**: Outputs the input directly if positive; otherwise, outputs zero. Helps mitigate the vanishing gradient problem.
-**LogSoftmax**: Applies the logarithm of the softmax function, useful with the Negative Log Likelihood Loss for classification.

## Training the Model
```bash
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
    
    training_time = (time() - time0) / 60
    print(f"\nTraining Time (in minutes) = {training_time}")
    return model, training_time
```
**Explanation**:

**Function Purpose**: Trains the neural network model using the training data.

**Parameters**:

-**model**: The neural network to be trained.
-**trainloader**: DataLoader containing the training dataset.
-**epochs**: Number of times the entire dataset is passed through the network.
-**learning_rate**: Determines the step size during weight updates.
-**momentum**: Helps accelerate gradients in the right direction, leading to faster convergence.

**Training Process**:

-**Device Configuration**: Utilizes GPU (cuda) if available for faster computation; otherwise, defaults to CPU.
-**Loss Function (Criterion)**: nn.NLLLoss is suitable for multi-class classification tasks with log-probabilities.
-**Optimizer**: optim.SGD (Stochastic Gradient Descent) with momentum accelerates the training process.

**Training Loop**:

-**Epoch Loop**: Iterates over the specified number of epochs.

**Batch Processing**:
-**Reshaping Images**: Flattens the 28x28 images into a 784-dimensional vector to match the input layer.
-**Zero Gradients**: Clears old gradients to prevent accumulation.
-**Forward Pass**: Computes the model's output.
-**Loss Calculation**: Measures the discrepancy between predicted and actual labels.
-**Backward Pass**: Computes gradients of the loss with respect to model parameters.
-**Optimizer Step**: Updates model parameters based on computed gradients.
-**Loss Tracking**: Accumulates loss over batches to monitor training progress.
-**Training Time**: Calculates the total time taken to train the model in minutes.

**Underlying Principles**:

-**Stochastic Gradient Descent (SGD)**: An optimization algorithm that updates model parameters incrementally using small batches, leading to faster convergence and reduced computational load.
-**Momentum**: Enhances SGD by considering past gradients to smooth out updates, helping to navigate ravines and avoid oscillations.
-**Backpropagation**: A fundamental algorithm for training neural networks, efficiently computes gradients by propagating errors backward through the network.

## Evaluating the Model
```bash
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
    
    accuracy = correct_count / all_count
    print(f"Number Of Images Tested = {all_count}")
    print(f"Model Accuracy = {accuracy}")
    return accuracy
```
**Explanation**:

**Function Purpose**: Assesses the trained model's performance on the validation dataset by calculating its accuracy.

**Parameters**:

-**model**: The trained neural network.
-**valloader**: DataLoader containing the validation dataset.

**Evaluation Process**:

-**Device Configuration**: Ensures the model runs on the appropriate device (GPU or CPU).
-**Iterating Over Validation Data**:
-**Reshaping Images**: Flattens each image to match the input layer.
-**Disabling Gradient Calculation**: torch.no_grad() saves memory and computation since gradients aren't needed during evaluation.
-**Forward Pass**: Computes the model's output.
-**Probability Conversion**: Transforms log-probabilities back to probabilities using the exponential function.
-**Prediction**: Identifies the class with the highest probability.
-**Accuracy Calculation**: Compares the predicted label with the true label, tallying correct predictions.
-**Result**: Prints and returns the model's accuracy on the validation dataset.

**Underlying Principles**:

-**Model Evaluation**: Critical for understanding how well the model generalizes to unseen data.
-**Accuracy Metric**: Represents the proportion of correct predictions out of the total predictions, providing a straightforward measure of performance.
-**Gradient-Free Evaluation**: Disabling gradient computations during evaluation optimizes memory usage and speeds up the process.

## Main Function
```bash
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
**Explanation**:

**Function Purpose**: Serves as the entry point of the script, coordinating the workflow from data loading to model evaluation.

**Workflow Steps**:

-**Data Loading**: Retrieves the training and validation datasets.
-**Data Visualization**: Displays a sample of images to understand the data better.
-**Model Creation**: Builds the neural network architecture.
-**Model Training**: Trains the model using the training dataset.
-**Model Evaluation**: Assesses the trained model's performance on the validation dataset.
-**Execution Control**: The if __name__ == "__main__": block ensures that main() is called only when the script is run directly, not when imported as a module.

**Underlying Principles**:

**Modular Programming**: Breaking down the code into functions enhances readability, maintainability, and reusability.
**Orchestrating Workflow**: The main function ties together various components, ensuring a logical and sequential execution flow.

## Results
Upon successful execution, the script will output the training loss for each epoch and the final model accuracy on the validation dataset. An example output might look like:
```bash
Epoch 0 - Training loss: 0.6931
Epoch 1 - Training loss: 0.6929
...
Epoch 14 - Training loss: 0.0345

Training Time (in minutes) = 0.5
Number Of Images Tested = 10000
Model Accuracy = 0.98
```
**This indicates that the model achieved an accuracy of 98% on the validation dataset, demonstrating effective learning and generalization.**



## Optimization(DIY!)

**Option 1 Adding Hidden Layers**:
You can enhance the model's expressive power by adding more hidden layers or increasing the number of neurons in each layer. For example:

```bash
model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], hidden_sizes[2]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[2], output_size),
    nn.LogSoftmax(dim=1)
)
```

**Option 2 Using Other Activation Functions**:
In addition to ReLU, you can experiment with other activation functions such as LeakyReLU, ELU, Tanh, etc., to improve the model's performance and training stability.

```bash
model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.LeakyReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.LeakyReLU(),
    nn.Linear(hidden_sizes[1], output_size),
    nn.LogSoftmax(dim=1)
)
```

**Option 3 Introducing Regularization**:
Dropout: Add Dropout layers between hidden layers to prevent overfitting.

```bash
model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(hidden_sizes[1], output_size),
    nn.LogSoftmax(dim=1)
)

```

**Option 4 Batch Normalization**:
Add Batch Normalization layers after activation functions to accelerate the training process and improve the model's performance.

```bash
model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.BatchNorm1d(hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.BatchNorm1d(hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], output_size),
    nn.LogSoftmax(dim=1)
)

```

**Option 5 Learning Rate Scheduling**:
Use learning rate schedulers (such as StepLR, ReduceLROnPlateau) to dynamically adjust the learning rate, thereby improving training effectiveness.
```bash
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(epochs):
    # Training code
    ...
    scheduler.step()
```

**Option 5 Learning Rate Scheduling**:
Experiment with different optimizers like SGD with Momentum, RMSprop, etc., to compare the impact of various optimization strategies on model training.
```bash
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

```
