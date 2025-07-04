from typing import List

import torch.nn as nn


def compute_receptive_field(layers: List[nn.Module]) -> int:
    """
    Вычисляет эффективное рецептивное поле (receptive field) для заданной последовательности свёрточных и пуллинговых слоёв.
    :param layers: Список слоёв нейронной сети (обычно последовательных), для которых нужно вычислить рецептивное поле.
    :return: Размер рецептивного поля (в пикселях) по одной размерности (ширина или высота).
    """
    rf = 1
    jump = 1
    for layer in layers:
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            k = layer.kernel_size
            s = layer.stride if hasattr(layer, 'stride') else 1
            k = k[0] if isinstance(k, tuple) else k
            s = s[0] if isinstance(s, tuple) else s
            rf = rf + (k - 1) * jump
            jump = jump * s
    return rf
