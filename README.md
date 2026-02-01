# Classification-cifar10_cnn

## Overview
This project implements a **Convolutional Neural Network (CNN)** for **multi-class image classification** using the **CIFAR-10 dataset** and **PyTorch**.  
The model consists of a simple two-layer architecture with:
- One convolutional (CNN) layer for feature extraction
- One fully connected (linear) layer for classification

The goal is to classify images into one of **10 categories**.

---

## Dataset
- **Name**: CIFAR-10  
- **Image size**: 32 × 32 pixels, RGB  
- **Training samples**: 50,000  
- **Test samples**: 10,000  
- **Number of classes**: 10  

The dataset is loaded locally using `torchvision.datasets.CIFAR10`.

---

## Model Architecture
1. **Convolutional Layer**: 3 input channels, 16 output channels, kernel size 3×3, padding 1  
2. **Activation**: ReLU  
3. **Pooling**: Max Pooling with 2×2 kernel and stride 2  
4. **Fully Connected Layer**: Linear layer mapping flattened features to 10 output classes

---

## Training Details
- **Framework**: PyTorch  
- **Loss function**: CrossEntropyLoss  
- **Optimizer**: Adam  
- **Learning rate**: 0.001  
- **Batch size**: 64  
- **Epochs**: 5  

---

## Training Results
- Epoch 1, Loss: 1.4696
- Epoch 2, Loss: 1.2085
- Epoch 3, Loss: 1.1118
- Epoch 4, Loss: 1.0418
- Epoch 5, Loss: 0.9918
- Test Accuracy: 62.01%



