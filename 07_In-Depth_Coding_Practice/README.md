### Coding iterations to hit the following targets on MNIST data
1. 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
2. Less than or equal to 15 Epochs
3. Less than 8000 Parameters

### A walkthrough of the target, results and analysis of each of the 4 iterations

#### Iteration 1
<b>Target :</b> To build the basic skeletion ensuring 1. receptive field close to image size i.e 28 2. Parameters < 8k 3. Correct placement of max pooling to get edges,gradients and textures,patterns<br>
<b>Results :</b> The final model architecture is C->C->T->C->C->T->C->1x1->GAP with number of channels = 16 across all layers<br>
<b>Analysis :</b> While there isnt such a huge gap between train and test accuracies, there is scope for learning. And since I cannot increase the parameters by a lot, I went for Batch Normalization next to make the model train harder

#### Iteration 2
<b>Target :</b> To improve the overall perfromance of the model by including batch normalization<br>
<b>Results :</b> The train and test accuracies have hit the 99% benchmark<br>
<b>Analysis :</b> While the test and training accuracies have both increased, there is a possibility of overfitting in the last few epochs. I will try dopout next to address this issue

#### Iteration 3
<b>Target :</b> To reduce overfitting by including dropout = 0.1<br>
<b>Results :</b> The gap between train and test accuracies has reduced but the overall performance has fallen to 98%<br>
<b>Analysis :</b> To improve test accuracy, I have to improve the train accuracy by improving the learning capacity of the model. I am going to try 1. Increasing the number of kernels if possible 2. Add image augmentation 3. trying a step learning scheduler

#### Iteration 4
<b>Target :</b> To improve the overall perfromance of the model by increasing number of weights, adding image augmentation and using a step LR to try to hit the target accuracy faster<br>
<b>Results :</b> The train and test accuracies are fluctuating about 98.8%<br>
<b>Analysis :</b> Increasing the weights(by +300) and image augmentation(random rotation and random affine) has not helped improve the train accuracy for some reason. Therefore step LR doesnt seem to help much either. YTD : I would still try to change the model architecture to add more weight(by adding kernels), probably not have a sencond transition block and instead have only 2 convolutions after the first transition block. 