import torch
from torch import nn


class VGG16DoubleHeadClassifier(nn.Module):
    def __init__(self):
        super(VGG16DoubleHeadClassifier, self).__init__()

        self.abnormal_head = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1),
            nn.Sigmoid(),
        )
        self.tuberculosis_head = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        abnormal = self.abnormal_head(x)
        tuberculosis = self.tuberculosis_head(x)

        return torch.cat((abnormal, tuberculosis), dim=1).type(torch.float)


class VGG16SingleHeadClassifier(nn.Module):
    def __init__(self):
        super(VGG16SingleHeadClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(x)
