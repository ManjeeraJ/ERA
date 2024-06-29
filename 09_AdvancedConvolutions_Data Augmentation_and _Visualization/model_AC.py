import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    A convolutional neural network (CNN) for image classification.

    This network is designed to process grayscale images (single channel) 
    and classify them into one of 10 categories. The architecture consists 
    of several convolutional blocks, transition blocks, pooling layers, 
    and an output block that uses global average pooling.

    """
    def __init__(self):
        super(Net, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),  
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=2, stride = 2, dilation = 2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32)        

        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),  
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=2, stride = 2, dilation = 2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)        

        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),  
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=2, stride = 2, dilation = 2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)        

        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1, groups=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1) , padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128)        

        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = self.output(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)