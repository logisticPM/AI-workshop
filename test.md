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
Explanation:

Function Purpose: Loads the MNIST dataset, downloading it if it's not already present locally. It returns DataLoaders for both training and validation datasets.

Parameters:

- **data_dir**: Specifies where to store or load the MNIST data.
- **batch_size**: Determines the number of samples per batch during training and validation.

Data Transformation:

- **ToTensor**: Converts PIL images or NumPy arrays into PyTorch tensors.
- **Normalize**: Standardizes the dataset by adjusting the mean and standard deviation. Here, both are set to 0.5 for grayscale images.
- 
Dataset Downloading:

Checks if the MNIST data exists locally. If not, it downloads the dataset.
train=True loads the training set, while train=False loads the validation set.

DataLoaders:

DataLoader wraps the dataset and provides batching, shuffling, and parallel data loading.
shuffle=True ensures the data is randomized, enhancing the model's ability to generalize.

Underlying Principles:

- **Data Normalization**: Scaling input data helps in faster convergence and stabilizes the training process by ensuring that all input features contribute equally.
- **Batching**: Processing data in batches improves computational efficiency and allows the model to update its weights more frequently.
