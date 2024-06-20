from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
from torchsummary import summary
from tqdm import tqdm

def getTrainTransforms():
    # Train data transformations
    # Apply a series of transformations to the training data
    return transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),  # Resize the image to 28x28 pixels
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),   # Convert the image to PyTorch tensor and scales it between 0 and 1
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize the tensor with mean and standard deviation(of the complete dataset). Also it is a tuple of means across each channel
        ])

def getTestTransforms():
    # Test data transformations
    # Apply transformations to the test data
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

def getDataLoader(batch_size = 512):
    """
    This function loads the MNIST dataset from the specified directory, applies the transformations,
    and returns the DataLoader for both the train and test datasets.

    Args:
    batch_size (int, optional): The number of samples per batch. Defaults to 512.

    Returns:
    tuple: The DataLoader objects for the train and test datasets.
    """
   
    # Load the MNIST dataset for training, apply transformations and download if not present
    train_data = datasets.MNIST('../data', train=True, download=True, transform=getTrainTransforms())
    
    # Load the MNIST dataset for testing, apply transformations and download if not present
    test_data = datasets.MNIST('../data', train=False, download=True, transform=getTestTransforms())

    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    # Create a DataLoader for the test dataset
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    
    # Create a DataLoader for the train dataset
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    
    return train_loader, test_loader

# The function definition should first list all required parameters, 
# followed by any optional parameters with default values.
def getSampleImages(loader, num_images = 10):
    """
    Retrieves a sample of images from the given DataLoader object.

    Args:
    loader (DataLoader): The DataLoader object from which to retrieve images.
    num_images (int, optional): The number of images to retrieve. Defaults to 10.

    Returns:
    None
    """

    # Get the first batch of images and labels from the DataLoader
    batch_data, batch_label = next(iter(loader)) 

    fig = plt.figure()

    for i in range(num_images):
      plt.subplot(3,4,i+1)
      plt.imshow(batch_data[i].squeeze(0), cmap='gray')
      plt.title(batch_label[i].item())
      plt.xticks([])
      plt.yticks([])
        
    plt.tight_layout()
    plt.show();  # Semicolon to suppress automatic display

def getModelSummary(model):
    """
    Prints a detailed summary of a PyTorch model.

    Parameters:
    model (torch.nn.Module): The PyTorch model to summarize.

    Returns:
    None
    """
    # Print the model summary
    summary(model, input_size=(1, 28, 28))

def getCorrectPredCount(predictions, labels):
    """
    Calculate the number of correct predictions.

    This function takes the model's predictions and the true labels, compares them to determine
    how many predictions are correct, and returns the total count of correct predictions.

    Parameters:
    predictions (torch.Tensor): The predictions made by the model.
    labels (torch.Tensor): The true labels.

    Returns:
    int: The number of correct predictions.
    """
    return predictions.argmax(dim=1).eq(labels).sum().item()

def train(model, device, train_loader, optimizer, criterion, train_losses, train_acc):
    """
    Perform one epoch of training on the given model.

    This function trains the model using the provided training data loader for one epoch.
    It performs forward and backward passes, computes the loss, updates the model parameters,
    and tracks the training loss and accuracy.

    Parameters:
    model (torch.nn.Module): The neural network model to be trained.
    device (torch.device): The device (CPU/GPU) to run the training on.
    train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
    optimizer (torch.optim.Optimizer): The optimizer to adjust the model's weights.
    train_acc (list): A list to store the training accuracy for each epoch.
    train_losses (list): A list to store the training loss for each epoch.

    Returns:
    None
    """
    model.train()  # Set the model to training mode
    pbar = tqdm(train_loader)  # Initialize a progress bar

    train_loss = 0  # Initialize the total training loss
    correct = 0  # Initialize the count of correct predictions
    processed = 0  # Initialize the count of processed samples

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)  # Move data and labels to the specified device
        optimizer.zero_grad()  # Clear the gradients from the previous iteration

        # Predict
        pred = model(data)  # Perform a forward pass to get predictions

        # Calculate loss
        loss = criterion(pred, target)  # Compute the negative log-likelihood loss
        train_loss += loss.item()  # Accumulate the training loss

        # Backpropagation
        loss.backward()  # Compute the gradients
        optimizer.step()  # Update the model parameters

        correct += getCorrectPredCount(pred, target)  # Count the correct predictions
        processed += len(data)  # Count the processed samples

        # Update the progress bar with current loss and accuracy
        pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    # Log training accuracy and loss for the epoch
    train_acc.append(100 * correct / processed)  # Record the training accuracy
    train_losses.append(train_loss / len(train_loader))  # Record the average training loss

def test(model, device, test_loader, criterion, test_losses, test_acc):
    """
    Evaluate the model on the test dataset.

    This function tests the model using the provided test data loader.
    It computes the loss and accuracy of the model on the test set and appends these metrics to the provided lists.

    Parameters:
    model (torch.nn.Module): The neural network model to be evaluated.
    device (torch.device): The device (CPU/GPU) to run the evaluation on.
    test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
    test_acc (list): A list to store the test accuracy for each epoch.
    test_losses (list): A list to store the test loss for each epoch.

    Returns:
    None
    """
    model.eval()  # Set the model to evaluation mode

    test_loss = 0  # Initialize the total test loss
    correct = 0  # Initialize the count of correct predictions

    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)  # Move data and labels to the specified device

            output = model(data)  # Perform a forward pass to get predictions
            test_loss += criterion(output, target, reduction='sum').item()  # Sum up batch loss

            correct += getCorrectPredCount(output, target)  # Count the correct predictions

    test_loss /= len(test_loader.dataset)  # Compute average test loss
    test_acc.append(100. * correct / len(test_loader.dataset))  # Record the test accuracy
    test_losses.append(test_loss)  # Record the test loss

    # Print the test results
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def training(model, device, num_epochs, train_loader, test_loader, optimizer, criterion, scheduler):
    """
    This function orchestrates the entire training process for a given model over a specified number of epochs.
    It handles training, testing, and learning rate adjustments, while keeping track of the losses and accuracies.
  
    Parameters:
    model (torch.nn.Module): The neural network model to be trained and tested.
    device (torch.device): The device (CPU/GPU) to run the training and testing on.
    num_epochs (int): The number of times to iterate over the entire training dataset.
    train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
    test_loader (torch.utils.data.DataLoader): The data loader for the testing dataset.
    optimizer (torch.optim.Optimizer): The optimizer to adjust the model's weights.
    criterion (torch.nn.Module): The loss function to evaluate the model's predictions.
    scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to adjust the learning rate during training.

    Returns:
    tuple: Containing lists of training losses, test losses, training accuracies and test accuracies
    - train_losses (list): Training loss recorded at each epoch.
    - test_losses (list): Test loss recorded at each epoch.
    - train_acc (list): Training accuracy recorded at each epoch.
    - test_acc (list): Test accuracy recorded at each epoch.
    """
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}')
        train(model, device, train_loader, optimizer, criterion, train_losses, train_acc)
        test(model, device, test_loader, criterion, test_losses, test_acc)
        scheduler.step()  # Checks if learning rate should reduce based on the criterion we mentioned when defining scheduler. In this case, after 15 epochs it reduces
    return train_losses, test_losses, train_acc, test_acc

def getTrainingTestPlots(train_losses, test_losses, train_acc, test_acc):
    """Generates and displays plots for training and test losses and accuracy.

    Args:
        train_losses (list): A list of training losses.
        test_losses (list): A list of test losses.
        train_acc (list): A list of training accuracy values.
        test_acc (list): A list of test accuracy values.

    Returns:
        None
    """
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
