import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    A convolutional neural network (CNN) for image classification.

    This network is designed to process grayscale images (single channel) 
    and classify them into one of 10 categories. The architecture consists 
    of several convolutional blocks, transition blocks, pooling layers, 
    and an output block that uses global average pooling.

    Attributes:
    -----------
    convblock1 : nn.Sequential
        Input block with a convolutional layer followed by ReLU activation.
    convblock2 : nn.Sequential
        Convolution Block 1 with a convolutional layer followed by ReLU activation.
    convblock3 : nn.Sequential
        Transition Block 1 with a 1x1 convolutional layer.
    pool1 : nn.MaxPool2d
        Max pooling layer in Transition Block 1.
    convblock4 : nn.Sequential
        Convolution Block 2 with a convolutional layer followed by ReLU activation.
    convblock5 : nn.Sequential
        Convolution Block 3 with a convolutional layer followed by ReLU activation.
    convblock6 : nn.Sequential
        Transition Block 2 with a 1x1 convolutional layer.
    pool2 : nn.MaxPool2d
        Max pooling layer in Transition Block 2.
    convblock7 : nn.Sequential
        Convolution Block 4 with a convolutional layer (with padding) followed by ReLU activation.
    gap : nn.Sequential
        Global average pooling layer in the Output Block.
    convblock8 : nn.Sequential
        Final 1x1 convolutional layer in the Output Block.
    """
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )  # r_out = 3, output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )  # r_out = 5, output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        )  # r_out = 5, output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2)  # r_out = 6, output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )  # r_out = 10, output_size = 10

        # CONVOLUTION BLOCK 3
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )  # r_out = 14, output_size = 8

        # TRANSITION BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        )  # r_out = 14, output_size = 8
        self.pool2 = nn.MaxPool2d(2, 2)  # r_out = 15, output_size = 4

        # CONVOLUTION BLOCK 4
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )  # r_out = 23, output_size = 4

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        )  # output_size = 1        

        # 1X1
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Parameters:
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 28, 28), representing a batch of grayscale images.

        Returns:
        torch.Tensor
            Output tensor of shape (batch_size, 10) with log probabilities for each class.
        """
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.gap(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)