import torch
from torch import nn


class SingleLayerClassifier(nn.Module):
    def __init__(self, input_size, num_labels, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(input_size, num_labels)

    def forward(self, pooled_output):
        x = self.dropout(pooled_output)
        return self.classifier(x)