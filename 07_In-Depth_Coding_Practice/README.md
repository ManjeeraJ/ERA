# 📚 Session 7 Assignment

## 📌 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Introduction](#introduction)
3. [Code Iterations](#file-structure)
4. [Additional Resources](#additional-resources)

## 🎯 Problem Statement

1. Your new target is:
    - 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
    - Less than or equal to 15 Epochs
    - Less than 8000 Parameters
    - Do this using your modular code. Every model that you make must be there in the model.py file as Model_1, Model_2, etc.
2. Do this in exactly 3 steps
3. Each File must have a "target, result, analysis" TEXT block (either at the start or the end)
4. You must convince why have you decided that your target should be what you have decided it to be, and your analysis MUST be correct. 
5. Evaluation is highly subjective, and if you target anything out of the order, marks will be deducted. 
6. Explain your 3 steps using these targets, results, and analysis with links to your GitHub files (Colab files moved to GitHub). 
7. Keep Receptive field calculations handy for each of your models. 
8. If your GitHub folder structure or file_names are messy, -100. 
9. When ready, attempt SESSION 7 -Assignment Solution

### Coding iterations to hit the following targets on MNIST data
1. 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
2. Less than or equal to 15 Epochs
3. Less than 8000 Parameters

## 📚 Introduction

The goal of the assignment is to iteratively reach the target accuracy by systematically making changes in the code (Model architecture, Image augmentation, Learning rates etc). Changes have to be made based on the performance of the training logs.

## 🔄 Code Iterations

### ⭐ [Iteration 1](./Iteration_1.ipynb)
**<u>Target:</u>**
- Build the basic skeleton ensuring:
  1. Receptive field close to image size (28).
  2. Parameters < 8k.
  3. Correct placement of max pooling to get edges, gradients, textures, and patterns.

**<u>Results:</u>**
- Final model architecture: `C -> C -> T (1x1 -> MP) -> C -> C -> T (1x1 -> MP) -> C -> GAP -> 1x1`, with 16 channels across all layers. Note: 1x1 can occur after MP in the Transition block to reduce memory during multiplication.  
  - **Best train accuracy:** 98.55%  
  - **Best test accuracy:** 98.57%

**<u>Analysis:</u>**
- The gap between train and test accuracies is small, indicating potential for further learning. To enhance learning without significantly increasing parameters, Batch Normalization will be introduced next.

### 🌟 [Iteration 2](./Iteration_2.ipynb)
**<u>Target:</u>**
- Improve overall performance by including Batch Normalization.

**<u>Results:</u>**
- Train and test accuracies have surpassed the 99% benchmark.  
  - **Best train accuracy:** 99.50%  
  - **Best test accuracy:** 99.12%

**<u>Analysis:</u>**
- Both accuracies have increased, but potential overfitting is observed in the final epochs. To address this, dropout will be introduced in the next iteration.

### ✨ [Iteration 3](./Iteration_3.ipynb)
**<u>Target:</u>**
- Reduce overfitting by including dropout (0.1).

**<u>Results:</u>**
- The gap between train and test accuracies has reduced, but overall performance has fallen to around 98%.  
  - **Best train accuracy:** 98.85%  
  - **Best test accuracy:** 99.17%

**<u>Analysis:</u>**
- To improve test accuracy, train accuracy must be enhanced by:  
  1. Increasing the number of kernels if possible.  
  2. Adding image augmentation.  
  3. Trying a step learning scheduler.

### 💫 [Iteration 4](./Iteration_4.ipynb)
**<u>Target:</u>**
- Improve overall performance by increasing the number of weights, adding image augmentation, and using a step LR to hit the target accuracy faster.

**<u>Results:</u>**
- Train and test accuracies are fluctuating around 98.8%.  
  - **Best train accuracy:** 98.87%  
  - **Best test accuracy:** 99.15%

**<u>Analysis:</u>**
- Increasing weights (by +300) and adding image augmentation (random rotation and random affine) haven't improved train accuracy significantly. The step LR also hasn't been very effective. Further actions to improve accuracy include:  
  1. Changing the model architecture to add more weights (by adding kernels).  
  2. Possibly eliminating the second transition block and having only 2 convolutions after the first transition block.

## 📖 Additional Resources
1. We need to calculate the mean and standard deviation across each channel for full dataset. The code can be found [here](./ERA1S7F1.ipynb).