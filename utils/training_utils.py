import time
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Union[torch.device, str]
) -> Tuple[float, float, float]:
    """
    Выполняет одну эпоху обучения модели.
    :param model: Обучаемая модель.
    :param loader: Даталоадер, возвращающий батчи входов и меток.
    :param criterion: Функция потерь (например, nn.CrossEntropyLoss).
    :param optimizer: Оптимизатор (например, Adam, SGD).
    :param device: Устройство для вычислений (cpu или cuda).
    :return: Tuple[Среднее значение функции потерь за эпоху, Средняя точность (accuracy) за эпоху, Время выполнения эпохи в секундах]:
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    start = time.time()
    for inputs, targets in tqdm(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    return running_loss/total, correct/total, time.time()-start


def eval_model(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: Union[torch.device, str]
) -> Tuple[float, float, float]:
    """
    Выполняет один проход по датасету (валидация или тестирование) без обновления весов.
    :param model: Модель, которую нужно оценить.
    :param loader: Даталоадер, возвращающий батчи входов и меток.
    :param criterion: Функция потерь.
    :param device: Устройство для вычислений (cpu или cuda).
    :return: Tuple[Среднее значение функции потерь за проход, Средняя точность (accuracy) за проход, Время выполнения в секундах]:
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    start = time.time()
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return running_loss/total, correct/total, time.time()-start
