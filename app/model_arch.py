import torch.nn as nn


class CareerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CareerClassifier, self).__init__()

        # Layer 1: 27 -> 512
        self.layer1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Layer 2: 512 -> 256
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        # Output Layer: 256 -> num_classes
        self.output_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        # Layer 1
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Layer 2
        out = self.layer2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Output
        out = self.output_layer(out)
        return out
