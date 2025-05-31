import torch.nn as nn
import torch.nn.functional as F


class BinaryCNN(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(BinaryCNN, self).__init__()

        # Блок 1: (224x224x3) → (112x112x32)
        # Сверточный слой с 32 фильтрами размером 3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Блок 2: (112x112x32) → (56x56x64)
        # Увеличиваем количество каналов до 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Блок 3: (56x56x64) → (28x28x128)
        # Финальная свертка с увеличением глубины до 128 каналов
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Пулинг для уменьшения пространственных размеров в 2 раза
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Глобальный пулинг: (28x28x128) → (1x1x128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Регуляризация для предотвращения переобучения
        self.dropout = nn.Dropout(dropout_prob)

        # Полносвязный слой для бинарной классификации
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (batch, 128)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class BinaryImprovedCNN(nn.Module):
    """
    Улучшенная версия CNN для бинарной классификации:
    - Добавлен дополнительный сверточный блок (28x28x128 → 14x14x256)
    - Использован LeakyReLU вместо ReLU для решения проблемы "мертвых" нейронов
    - Увеличено количество параметров с 128 до 256 в финальном слое
    """

    def __init__(self, dropout_prob=0.3):  # уменьшен dropout для улучшения обучения
        super(BinaryImprovedCNN, self).__init__()

        # Блок 1: (224x224x3) → (112x112x32)
        # Начальная обработка изображения с 32 фильтрами
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Блок 2: (112x112x32) → (56x56x64)
        # Удвоение количества каналов
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Блок 3: (56x56x64) → (28x28x128)
        # Дальнейшее увеличение глубины признаков
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Блок 4: (28x28x128) → (14x14x256)
        # Дополнительный слой для извлечения более сложных признаков
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Пулинг с шагом 2 для уменьшения размерности
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Глобальный пулинг: (14x14x256) → (1x1x256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Регуляризация для предотвращения переобучения
        self.dropout = nn.Dropout(dropout_prob)

        # Финальный классификатор
        self.fc = nn.Linear(256, 1)

        # LeakyReLU вместо ReLU
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        x = self.pool(self.activation(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (batch, 256)
        x = self.dropout(x)
        x = self.fc(x)
        return x
