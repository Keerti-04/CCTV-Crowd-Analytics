# csrnet.py
import torch
import torch.nn as nn
from torchvision import models


class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        reduced_channels = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class CSRNet(nn.Module):
    """
    CSRNet: Crowd Counting via Convolutional Neural Network.
    Frontend: VGG16 conv layers up to conv5_3
    Backend: dilated conv layers
    """
    def __init__(self):
        super(CSRNet, self).__init__()
        
        
        vgg_features = list(models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.children())
        self.frontend = nn.Sequential(*vgg_features[:23])  

        # SE block inserted after the frontend
        self.se = SEBlock(512)
        
        # Backend: dilated convolution layers
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        
        # Output layer (1-channel density map)
        self.output_layer = nn.Conv2d(64, 1, 1)
    
    def forward(self, x):
        x = self.frontend(x)
        x = self.se(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    # Quick test
    model = CSRNet()
    print("CSRNet model initialized successfully.")
    x = torch.randn(1, 3, 640, 480)
    y = model(x)
    print(f"Output density map shape: {y.shape}")