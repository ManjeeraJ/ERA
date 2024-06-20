# 📚 Session 4 Assignment

## 📌 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Introduction](#introduction)
3. [File Structure](#file-structure)

## 🎯 Problem Statement

1. Re-look at the code that we worked on in Assignment 4 (the fixed version). 
2. Move the contents of the code to the following files:
    - model.py
    - utils.py
    - S5.ipynb
3. Make the whole code run again. 
4. Upload the code with the 3 files + README.md file (total 4 files) to GitHub. README.md (look at the spelling) must have details about this code and how to read your code (what file does what). Heavy negative scores for not formatting your markdown file into p, H1, H2, list, etc. 
5. Attempt Assignment 5. 

## 📚 Introduction

The goal of the assignment is to reorganize the code blocks from Assignment 4 into separate files 1. model.py and 2. utils.py and ensure the code still runs successfully when calling the functions in S5.ipynb

## 📂 File Structure

- `model.py` :
    1. `class Net(nn.Module)`: This class defines a Convolutional Neural Network (CNN) architecture, which is a subclass of nn.Module. The network consists of several convolutional layers, followed by max-pooling layers, and then fully connected layers. It is designed to process and classify 2D image data.
- `utils.py` :
    1. `getTrainTransforms()`: Applies a series of transformations to the training data, including random center cropping, resizing, random rotation, tensor conversion, and normalization.
    2. `getTestTransforms()`: Applies transformations to the test data, specifically tensor conversion and normalization.
    3. `getDataLoader(batch_size=512)`: Loads the MNIST dataset, applies the respective transformations for train and test sets, and returns DataLoader objects for both.
    4. `getSampleImages(loader, num_images=10)`: Retrieves a sample of images from the given DataLoader and displays them using matplotlib
    5. `getModelSummary(model)`: Prints a detailed summary of a PyTorch model using the specified input size.
    6. `getCorrectPredCount(predictions, labels)`: Calculates and returns the number of correct predictions by comparing model outputs with true labels.
    7. `train(model, device, train_loader, optimizer, criterion, train_losses, train_acc)`: Trains the model for one epoch, updating the model parameters, and records the training loss and accuracy.
    8. `test(model, device, test_loader, criterion, test_losses, test_acc)`: Evaluates the model on the test dataset, recording the test loss and accuracy.
    9. `training(model, device, num_epochs, train_loader, test_loader, optimizer, criterion, scheduler)`: Orchestrates the entire training process over a specified number of epochs, handling training, testing, and learning rate adjustments.
    10. `getTrainingTestPlots(train_losses, test_losses, train_acc, test_acc)`: Generates and displays plots for training and test losses and accuracy using matplotlib.

- `S5.ipynb` : This is the main Jupyter notebook file that contains the code which uses the model and utility functions

