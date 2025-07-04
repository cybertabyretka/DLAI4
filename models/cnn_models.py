from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelSizeCNN(nn.Module):
    """
    Класс сверточной нейронной сети с настраиваемым размером ядра.
    """
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            kernel_size: int,
            base_channels: int,
            ref_kernel_size: int = 3
    ) -> None:
        """
        Функция инициализации класса свёрточной нейронной сети с настраиваемым размером ядра.
        :param in_channels: количество входных каналов.
        :param num_classes: количество выходных классов.
        :param kernel_size: размер сверточного ядра.
        :param base_channels: базовое количество каналов для расчета на основе эталонного ядра.
        :param ref_kernel_size: эталонный размер ядра для вычисления отношения каналов. По умолчанию 3.
        """
        super().__init__()

        ratio = (ref_kernel_size ** 2) / (kernel_size ** 2)

        channels = max(1, int(base_channels * ratio))

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, num_classes)

        self.channels = channels
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход сети.
        :param x: входной тензор формы (batch_size, in_channels, H, W)
        :return: выходной тензор формы (batch_size, num_classes)
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class MixedKernelCNN(nn.Module):
    """
    Сверточная нейросеть, использующая два ядра разного размера: 1x1 и 3x3.
    """
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            channels: int = 32
    ) -> None:
        """
        Инициализация слоев сети.
        :param in_channels: число входных каналов в изображении.
        :param num_classes: число классов для классификации.
        :param channels: число выходных каналов в сверточных слоях. По умолчанию 32.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть.
        :param x: входной тензор размерности (batch_size, in_channels, H, W).
        :return: выходной тензор размерности (batch_size, num_classes).
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class DepthCNN(nn.Module):
    """
    Сверточная нейросеть с динамическим числом слоев глубины.
    """
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            num_layers: int,
            channels: int = 32
    ) -> None:
        """
        Инициализация сети с заданным числом слоев.
        :param in_channels: количество каналов на входе.
        :param num_classes: количество классов для классификации.
        :param num_layers: количество сверточных слоев сети.
        :param channels: количество каналов во внутренних сверточных слоях. По умолчанию 32.
        """
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ))
            layers.append(nn.ReLU())
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через все слои.
        :param x: входной тензор размерности (batch_size, in_channels, H, W).
        :return: выходной тензор размерности (batch_size, num_classes).
        """
        for layer in self.layers:
            x = layer(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResBlock(nn.Module):
    """
    Остаточный блок (Residual Block) для сверточных сетей.
    """
    def __init__(self, channels: int) -> None:
        """
        Инициализация сверточных слоев остаточного блока.
        :param channels: количество каналов во входных и выходных тензорах.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через остаточный блок.

        Выполняется свертка, активация и добавление исходного тензора (skip connection).
        :param x: входной тензор размерности (batch_size, channels, H, W).
        :return: выходной тензор той же размерности.
        """
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        return self.relu(out)


class ResNetLike(nn.Module):
    """
    Простая ResNet-подобная архитектура с последовательностью остаточных блоков.
    """
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            num_blocks: int = 3,
            channels: int = 32
    ) -> None:
        """
        Инициализация ResNet-подобной модели.
        :param in_channels: количество каналов на входе.
        :param num_classes: количество классов для предсказания.
        :param num_blocks: число остаточных блоков в модели. По умолчанию 3.
        :param channels: число каналов в промежуточных сверточных слоях. По умолчанию 32.
        """
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, channels, 3, padding=1), nn.ReLU())
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock(channels))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через ResNet-подобную сеть.
        :param x: входной тензор размерности (batch_size, in_channels, H, W).
        :return: выходной тензор размерности (batch_size, num_classes).
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class GenericCNN(nn.Module):
    """
    Универсальная сверточная нейронная сеть для классификации изображений.
    """
    def __init__(
        self,
        input_channels: int = 1,
        input_size: Tuple[int, int] = (28, 28),
        num_classes: int = 10,
        conv_channels: Tuple[int, int] = (32, 64),
        linear_units: int = 128,
        dropout: float = 0.25
    ) -> None:
        """
        Инициализация слоев сверточной сети и вычисление размерности фич после сверточных слоев.
        :param input_channels: число каналов входного изображения. По умолчанию 1.
        :param input_size: размер входного изображения (H, W). По умолчанию (28, 28).
        :param num_classes: число выходных классов. По умолчанию 10.
        :param conv_channels: список чисел каналов для каждого сверточного слоя. По умолчанию (32, 64).
        :param linear_units: число скрытых нейронов в полносвязном слое. По умолчанию 128.
        :param dropout: доля зануления в Dropout. По умолчанию 0.25.
        """
        super().__init__()
        layers = []
        in_c = input_channels
        for out_c in conv_channels:
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2, 2))
            in_c = out_c
        self.conv_layers = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            out = self.conv_layers(dummy)
        feature_dim = out.numel()
        self.fc1 = nn.Linear(feature_dim, linear_units)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(linear_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сверточную и полносвязную части сети.
        :param x: входной тензор формы (batch_size, input_channels, H, W).
        :return: логиты предсказаний формы (batch_size, num_classes).
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class ResidualBlock(nn.Module):
    """
    Остаточный блок для ResNet-подобных архитектур.

    Структура блока:
        Conv-BatchNorm-ReLU -> Conv-BatchNorm -> Добавление пропуска (skip connection) -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """
        Инициализация слоев остаточного блока.
        :param in_channels: количество входных каналов.
        :param out_channels: количество выходных каналов.
        :param stride: шаг свертки в первом сверточном слое. По умолчанию 1.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через остаточный блок.
        :param x: входной тензор формы (batch_size, in_channels, H, W).
        :return: выходной тензор формы (batch_size, out_channels, H_out, W_out).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class GenericCNNWithResidual(nn.Module):
    """
    Универсальная сверточная сеть с встроенными остаточными блоками.
    """
    def __init__(
        self,
        input_channels: int = 1,
        input_size: Tuple[int, int] = (28, 28),
        num_classes: int = 10,
        channels: Tuple[int, int] = (32, 64),
        pool_out_size: Tuple[int, int] = (4, 4),
        dropout: float = 0.25
    ) -> None:
        """
        Инициализация слоев сети и вычисление размерности признаков для fc.
        :param input_channels: число каналов входного изображения.
        :param input_size: размер входного изображения (высота, ширина).
        :param num_classes: число выходных классов.
        :param channels: последовательность каналов для конволюций.
        :param pool_out_size: выходной размер пула.
        :param dropout: доля зануления в Dropout.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, channels[0], 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels[0])
        res_blocks = []
        in_c = channels[0]
        for c in channels:
            stride = 2 if in_c != c else 1
            res_blocks.append(ResidualBlock(in_c, c, stride))
            in_c = c
        self.residual_layers = nn.Sequential(*res_blocks)
        self.pool = nn.AdaptiveAvgPool2d(pool_out_size)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            x = F.relu(self.bn1(self.conv1(dummy)))
            x = self.residual_layers(x)
            x = self.pool(x)
        feature_dim = x.numel()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через весь GenericCNNWithResidual.
        :param x: входной тензор (batch_size, input_channels, H, W).
        :return: логиты (batch_size, num_classes).
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.residual_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
