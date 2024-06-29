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
        self.prep_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)     
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(128)     
        )

        self.resblock1 = nn.Sequential(
            ## conv block - 1
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
            padding=1, bias = False), # Input: 128X128X3 | Output:126  | RF: 6
            nn.ReLU(),
            nn.BatchNorm2d(128),

            ## conv block - 2
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
            padding=1, bias = False), # Input: 128X128X3 | Output:126  | RF: 6
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(256)     
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(512)     
        )

        self.resblock2 = nn.Sequential(
            ## conv block - 1
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
            padding=1, bias = False), # Input: 128X128X3 | Output:126  | RF: 6
            nn.ReLU(),
            nn.BatchNorm2d(512),

            ## conv block - 2
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
            padding=1, bias = False), # Input: 128X128X3 | Output:126  | RF: 6
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )

        self.pool = nn.MaxPool2d(4, 4)

        self.fc1 = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        # Prep Layer
        out = self.prep_block(x)
        # print(f"prep layer shape: {out.shape}")
        out = self.layer1(out)
        # print(f"layer1 layer shape: {out.shape}")

        res1 = self.resblock1(out)
        # print(f"res1 layer shape: {out.shape}")
        out = out + res1
        # print(f"outout after layer1 + res1 shape: {out.shape}")

        out = self.layer2(out)
        # print(f"layer2 layer shape: {out.shape}")

        out = self.layer3(out)
        # print(f"layer3 layer shape: {out.shape}")
        res2 = self.resblock2(out)
        # print(f"res2 layer shape: {out.shape}")
        out = out + res2
        # print(f"outout after layer3 + res2 layer shape: {out.shape}")

        out = self.pool(out)
        # print(f"pool layer shape: {out.shape}")

        out = out.view(-1, 512)
        # print(f"view shape: {out.shape}")

        out = self.fc1(out)
        # print(f"linear shape: {out.shape}")
        # out = out.view(-1, 10)
        return F.log_softmax(out, dim=1)  # out
        

