from typing import List

import torch
import torch.nn as nn


class FCNet(nn.Module):
    """
    Класс полносвязной модели.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int) -> None:
        """
        Функция инициализации класса полносвязной модели.
        :param input_dim: Размер входного вектора признаков.
        :param hidden_dims: Список размеров для каждого скрытого слоя.
        :param num_classes: Количество классов для классификации.
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход по полносвязной сети.
        :param x: Входной тензор
        :return: Выходной тензор
        """
        x = x.view(x.size(0), -1)
        return self.net(x)
