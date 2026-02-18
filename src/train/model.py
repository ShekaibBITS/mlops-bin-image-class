"""
Baseline CNN model for Cats vs Dogs binary classification.

Goal: simple architecture to satisfy baseline requirement.
"""

import torch  # Core PyTorch
import torch.nn as nn  # Neural network layers


class SimpleCNN(nn.Module):
    """
    A small CNN suitable as a baseline:
    - Conv -> ReLU -> Pool stacks
    - Flatten -> Linear -> Output logits for 2 classes
    """
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()  # Initialize nn.Module

        self.features = nn.Sequential(  # Feature extractor layers
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 3x224x224 -> 16x224x224
            nn.ReLU(),  # Non-linearity
            nn.MaxPool2d(2),  # 16x224x224 -> 16x112x112

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # -> 32x112x112
            nn.ReLU(),  # Non-linearity
            nn.MaxPool2d(2),  # -> 32x56x56

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> 64x56x56
            nn.ReLU(),  # Non-linearity
            nn.MaxPool2d(2),  # -> 64x28x28
        )

        self.classifier = nn.Sequential(  # Classification head
            nn.Flatten(),  # Flatten feature maps
            nn.Linear(64 * 28 * 28, 128),  # Fully connected layer
            nn.ReLU(),  # Non-linearity
            nn.Dropout(0.3),  # Regularization
            nn.Linear(128, num_classes),  # Output logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: features -> classifier -> logits."""
        x = self.features(x)  # Extract features
        x = self.classifier(x)  # Produce logits
        return x  # Return logits
    
