import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Defines a Convolutional Neural Network (CNN) architecture.

    The network architecture consists of several convolutional layers followed by
    max-pooling layers, and then fully connected layers. The architecture is designed
    to process and classify 2D image data.

    Layers:
        - Conv2d layers: Convolutional layers with ReLU activation and bias.
        - MaxPool2d layers: Max pooling layers to reduce the spatial dimensions.
        - Linear layers: Fully connected layers for classification.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        conv4 (nn.Conv2d): Fourth convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
    """

    def __init__(self):
        """
        Initializes the layers of the network.
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, bias=False)
        self.fc1 = nn.Linear(4096, 50, bias=False)  # Adjusted input features to the first fully connected layer
        self.fc2 = nn.Linear(50, 10, bias=False)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Output tensor after passing through the network layers.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 4096)  # Adjusted based on the size of the feature map after conv layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
