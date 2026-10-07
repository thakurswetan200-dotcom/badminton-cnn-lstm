import torch
import torch.nn as nn
from torchvision import models


class CNNLSTM(nn.Module):

    def __init__(self, num_classes=5):

        super(CNNLSTM, self).__init__()

        # CNN: ResNet18
        self.cnn = models.resnet18(weights=None)

        # Remove the original ResNet classifier
        self.cnn.fc = nn.Identity()

        # LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            batch_first=True
        )

        # Final classification layer
        self.fc = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x):

        # x shape:
        # [batch, sequence, channels, height, width]

        batch_size, sequence_length, channels, height, width = x.shape

        # Combine batch and sequence
        x = x.view(
            batch_size * sequence_length,
            channels,
            height,
            width
        )

        # CNN extracts spatial features
        x = self.cnn(x)

        # Restore sequence dimension
        x = x.view(
            batch_size,
            sequence_length,
            512
        )

        # LSTM learns temporal information
        x, _ = self.lstm(x)

        # Take the last time step
        x = x[:, -1, :]

        # Classification
        x = self.fc(x)

        return x