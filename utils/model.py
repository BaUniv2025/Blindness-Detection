import torch.nn as nn
import torch.nn.functional as F


class BinaryCNN(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(BinaryCNN, self).__init__()

        # (224, 224) → (112, 112)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # (112, 112) → (56, 56)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,
                               padding=1)  # (56, 56) → (28, 28)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # → (batch, 128, 1, 1)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(128, 1)  # выход — один логит для BCEWithLogitsLoss

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (batch, 128)
        x = self.dropout(x)
        x = self.fc(x)
        return x  # логиты (без сигмоиды)
