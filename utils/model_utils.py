import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """
    Считает количество обучаемых параметров модели.
    :param model: Модель.
    :return: Количество параметров.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)