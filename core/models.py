import torch
import torch.nn as nn
from torchvision import models

class ImprovedResNet50(nn.Module):
    """
    Fine-tuned ResNet50 model for Skin Lesion Classification.
    Freezes early layers, unfreezes layer4 for domain adaptation, and uses Dropout.
    """
    def __init__(self, num_classes):
        super(ImprovedResNet50, self).__init__()
        
        # Load pre-trained weights
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze entire network first
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Unfreeze layer4 to allow learning skin-specific complex textures
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
            
        # Replace the classifier head
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512), # Added BN for stability with MixUp
            nn.ReLU(inplace=True),
            nn.Dropout(0.4), # High dropout to prevent overfitting
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)
        
    def get_features(self, x):
        """Used for Grad-CAM"""
        # Manual forward pass to stop at layer4
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        features = self.resnet.layer4(x)
        return features


class SimpleCNN(nn.Module):
    """
    Baseline CNN provided for comparison purposes.
    Maintained identical to phase2/baseline_cnn.py
    """
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
