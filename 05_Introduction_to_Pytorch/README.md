
# Assignment

1. Re-look at the code that we worked on in Assignment 4 (the fixed version). 
2. Move the contents of the code to the following files:
    - model.py
    - utils.py
    - S5.ipynb
3. Make the whole code run again. 
4. Upload the code with the 3 files + README.md file (total 4 files) to GitHub. README.md (look at the spelling) must have details about this code and how to read your code (what file does what). Heavy negative scores for not formatting your markdown file into p, H1, H2, list, etc. 
5. Attempt Assignment 5. 

# Code breakdown and explanation

1. model.py contains
    - class Net - It defines the structure of the neural network and the forward function
    - model summary func - Returns the model summary given the model and input size
    - train and test funcs - They are called for every epoch and train accuracy, test accuracy, train loss, and test loss is stored
    - draw graphs func - Returns line plots of accuracies and losses mentioned above

2. utils.py contains
    - return dataset images func - returns the visualization of train images given the train loader and number of images to be seen
    - GetCorrectPredCount func - returns the count of correct predictions

3. Run S5.ipynb
    - Load libraries
    - Check CUDA availability
    - Define test and train loaders
    - Call return dataset images func from utils.py to visualize few images from train dataset
    - Call all functions from model.py to define the model and run model summary
    - Define optimizer and run epoch iterations calling train and test functions at every epoch
    - Plot the losses and accuracies
