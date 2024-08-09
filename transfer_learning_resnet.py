"""
PyTorch model class for transfer learning using a ResNet-50. The last layer outputs to two nodes for a binary classification task.

Author: Helene Li
"""

# Importing necessary libraries
import torch
from torchvision.models.resnet import ResNet50_Weights, resnet50


class TransferLearningResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # The model we want to transfer learn from
        weights = ResNet50_Weights.IMAGENET1K_V1
        self._model = resnet50(weights)

        # Freeze by setting requires_grad to false
        for parameters in self._model.parameters():
            parameters.requires_grad = False

        # Replace the last layer, changing outputs from 1000 to 2
        self._model.fc = torch.nn.Linear(2048, 2)

    def __call__(self, x):
        return self._model(x)
